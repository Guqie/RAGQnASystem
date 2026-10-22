import socket
import traceback

import streamlit as st

from agent.bert_ner import load_bert_aligner, recognizer_backend
from agent.config import settings
from agent.core import CounselingAgent
from agent.dictionary import DictionaryRecognizer, HybridRecognizer
from agent.knowledge import Neo4jKnowledge
from agent.llm import DeepSeekClient
from agent.memory import SessionMemory
from agent.prompts import offline_answer
from agent.sample_kg import demo_knowledge

MODELS = {
    "DeepSeek-Chat": "deepseek-chat",
    "DeepSeek-Reasoner": "deepseek-reasoner",
}


@st.cache_resource
def load_recognizer(entity_dir: str, root: str):
    base = DictionaryRecognizer(entity_dir)
    aligner = load_bert_aligner(root)
    if aligner is not None:
        return HybridRecognizer(base, aligner)
    return base


def connect_graph():
    if not settings.neo4j_password:
        return None, "还没有配置 NEO4J_PASSWORD，知识图谱检索会跳过。"
    try:
        import py2neo
    except Exception as exc:
        return None, f"未安装 py2neo：{exc}"
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(5)
    try:
        client = py2neo.Graph(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            name=settings.neo4j_database,
        )
        client.run("RETURN 1").data()
        return client, ""
    except Exception as exc:
        return None, str(exc)
    finally:
        socket.setdefaulttimeout(previous)


def ensure_windows():
    if "windows" not in st.session_state:
        st.session_state.windows = [{"messages": [], "memory": SessionMemory()}]
        st.session_state.active = 0
    windows = st.session_state.windows
    if not windows:
        windows.append({"messages": [], "memory": SessionMemory()})
        st.session_state.active = 0
    st.session_state.active = min(st.session_state.active, len(windows) - 1)
    return windows[st.session_state.active]


def render_trace(trace: list) -> None:
    for item in trace:
        st.markdown(f"**{item['stage']}** · {item['status']}")
        st.write(item.get("detail") or "")


def stream_answer(agent: CounselingAgent, turn, memory: SessionMemory, query: str, model_id: str) -> str:
    if turn.final_text is not None:
        st.markdown(turn.final_text)
        return turn.final_text
    placeholder = st.empty()
    parts = []
    try:
        for token in agent.llm.stream(turn.messages, model=model_id):
            parts.append(token)
            placeholder.markdown("".join(parts) + "▌")
        text = "".join(parts).strip()
        if not text:
            raise RuntimeError("模型没有返回内容")
        placeholder.markdown(text)
        return text
    except Exception as exc:
        text = offline_answer(
            query,
            memory,
            turn.knowledge,
            note=f"（大模型调用失败，已改为整理检索结果：{exc}）",
        )
        placeholder.markdown(text)
        return text


def main(is_admin, usname):
    st.set_page_config(page_title="心理咨询 Agent", layout="centered")
    st.title("心理咨询 Agent")
    st.caption("先判断要不要查知识图谱，再决定回答。支持多轮记忆，遇到危机表述会停止用药讨论。")

    with st.sidebar:
        logo = settings.root / "img" / "logo.jpg"
        if logo.exists():
            st.image(str(logo), use_column_width=True)
        st.caption(f"欢迎你，{'管理员' if is_admin else '用户'} {usname}。当前版本：2.0")

        ensure_windows()
        if st.button("新建对话窗口"):
            st.session_state.windows.append({"messages": [], "memory": SessionMemory()})
            st.session_state.active = len(st.session_state.windows) - 1
            st.rerun()

        labels = [f"对话窗口 {index + 1}" for index in range(len(st.session_state.windows))]
        selected = st.selectbox("请选择对话窗口", labels, index=st.session_state.active)
        st.session_state.active = labels.index(selected)

        model_name = st.selectbox("回答模型", list(MODELS))
        model_id = MODELS[model_name]
        show_trace = st.checkbox("显示 Agent 推理过程", value=True)
        use_demo_graph = st.checkbox("使用内置演示图谱", value=False)
        if use_demo_graph:
            st.info("演示图谱只包含抑郁症和焦虑症的样例，用来离线展示 Agent，不是完整医疗库。")

        if "graph_client" not in st.session_state:
            client, error = connect_graph()
            st.session_state.graph_client = client
            st.session_state.graph_error = error
        if st.session_state.graph_client is not None:
            st.success("知识图谱已连接")
        else:
            st.warning(st.session_state.graph_error or "知识图谱未连接")
        if st.button("重新连接知识图谱"):
            client, error = connect_graph()
            st.session_state.graph_client = client
            st.session_state.graph_error = error
            st.rerun()

        api_ready = bool(settings.deepseek_api_key)
        if api_ready:
            st.success("回答模型已配置")
        else:
            st.info("未配置 DEEPSEEK_API_KEY。系统仍会检索，并用检索结果整理回复。")

        if is_admin:
            st.markdown("---")
            st.caption("管理员可以到 Neo4j Browser 修改图谱。")
            st.markdown("[打开 Neo4j Browser](http://localhost:7474/browser/)")

        st.markdown("---")
        if st.button("退出登录"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.logged_in = False
            st.rerun()

    window = ensure_windows()
    memory = window["memory"]
    history = window["messages"]

    with st.spinner("正在准备实体词典..."):
        recognizer = load_recognizer(str(settings.entity_dir), str(settings.root))
    st.caption("实体识别：" + recognizer_backend(recognizer))

    llm = None
    if settings.deepseek_api_key:
        llm = DeepSeekClient(settings.deepseek_api_key, settings.deepseek_api_base, model_id)
    if use_demo_graph:
        knowledge = demo_knowledge()
    else:
        knowledge = Neo4jKnowledge(st.session_state.get("graph_client"))
    agent = CounselingAgent(knowledge, recognizer, llm)

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if show_trace and message.get("trace"):
                with st.expander("Agent 推理过程"):
                    render_trace(message["trace"])

    query = st.chat_input("想聊什么都可以，也可以问疾病、症状或用药")
    if not query:
        return

    history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("正在识别问题并检索知识图谱..."):
                turn = agent.step(query, memory)
            answer = stream_answer(agent, turn, memory, query, model_id)
            if show_trace:
                with st.expander("Agent 推理过程", expanded=True):
                    render_trace(turn.trace)
        except Exception:
            detail = traceback.format_exc()
            print(detail)
            answer = "处理这句话时出了问题。可以换种说法再试一次。"
            turn_trace = [{"stage": "错误", "status": "失败", "detail": "内部错误已记录在服务端日志。"}]
            st.error(answer)
            turn = None
        else:
            turn_trace = turn.trace

    if turn is not None:
        memory.observe(query, answer, turn)
    history.append({"role": "assistant", "content": answer, "trace": turn_trace})


if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True
        st.session_state.admin = True
        st.session_state.usname = "测试管理员"
    if st.session_state.get("logged_in", False):
        main(is_admin=st.session_state.get("admin", False), usname=st.session_state.get("usname", "用户"))
    else:
        st.error("请先登录。")
