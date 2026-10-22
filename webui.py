import socket
import traceback

import streamlit as st

from agent.bert_ner import load_bert_aligner, recognizer_backend
from agent.config import settings
from agent.core import AgentTurn, CounselingAgent
from agent.dictionary import DictionaryRecognizer, HybridRecognizer
from agent.knowledge import Neo4jKnowledge
from agent.llm import DeepSeekClient
from agent.memory import SessionMemory, load_session, memory_path, save_session
from agent.rag import DocumentStore, LazyDocuments
from agent.sample_kg import demo_documents, demo_knowledge
from agent.types import Perception

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


def ensure_windows(username: str):
    if "windows" not in st.session_state:
        windows, active = load_session(memory_path(settings.root, username))
        st.session_state.windows = windows
        st.session_state.active = active
    windows = st.session_state.windows
    if not windows:
        windows.append({"messages": [], "memory": SessionMemory()})
        st.session_state.active = 0
    st.session_state.active = min(st.session_state.active, len(windows) - 1)
    return windows[st.session_state.active]


def persist_windows(username: str) -> None:
    if "windows" not in st.session_state:
        return
    save_session(memory_path(settings.root, username), st.session_state.windows, st.session_state.active)


def build_documents(use_demo: bool):
    if use_demo:
        return demo_documents()
    source = settings.root / "data" / "medical_new_2.json"
    if not source.exists():
        return DocumentStore()
    cache = settings.root / "tmp_data" / "rag_index.pkl"
    return LazyDocuments(lambda: cached_store(source, cache))


def cached_store(source, cache):
    from agent.rag import cached_medical_store

    return cached_medical_store(str(source), str(cache))


def render_trace(trace: list) -> None:
    for item in trace:
        st.markdown(f"**{item['stage']}** · {item['status']}")
        st.write(item.get("detail") or "")


def consume_stream(agent: CounselingAgent, query: str, memory: SessionMemory, show_trace: bool):
    placeholder = st.empty()
    status_box = st.empty()
    parts = []
    trace = []
    done = {}
    for event in agent.stream(query, memory):
        kind = event["type"]
        if kind == "status":
            status_box.caption(event["text"])
        elif kind == "token":
            parts.append(event["text"])
            placeholder.markdown("".join(parts) + "▌")
        elif kind == "trace":
            trace.append(event["item"])
        elif kind == "done":
            done = event
    text = "".join(parts).strip() or "（没有生成回复）"
    placeholder.markdown(text)
    status_box.empty()
    if show_trace and trace:
        with st.expander("Agent 推理过程", expanded=True):
            render_trace(trace)
    turn = AgentTurn(
        final_text=text,
        messages=done.get("messages"),
        trace=trace,
        knowledge=done.get("knowledge") or "",
        plan=done.get("plan") or {},
        perception=done.get("perception") or Perception.empty(),
        resolved_diseases=list(done.get("resolved_diseases") or []),
        symptoms=list(done.get("symptoms") or []),
        intents=list(done.get("intents") or []),
        drug=done.get("drug"),
        crisis=bool(done.get("crisis")),
    )
    return text, turn


def main(is_admin, usname):
    st.set_page_config(page_title="心理咨询 Agent", layout="centered")
    st.title("心理咨询 Agent")
    st.caption("自然语言问题会同时查知识图谱和疾病文档，回答按片段流出。对话保留长上下文，更早的内容可以再检索。")

    with st.sidebar:
        logo = settings.root / "img" / "logo.jpg"
        if logo.exists():
            st.image(str(logo), use_column_width=True)
        st.caption(f"欢迎你，{'管理员' if is_admin else '用户'} {usname}。当前版本：3.0")

        ensure_windows(usname)
        if st.button("新建对话窗口"):
            st.session_state.windows.append({"messages": [], "memory": SessionMemory()})
            st.session_state.active = len(st.session_state.windows) - 1
            persist_windows(usname)
            st.rerun()

        labels = [f"对话窗口 {index + 1}" for index in range(len(st.session_state.windows))]
        selected = st.selectbox("请选择对话窗口", labels, index=st.session_state.active)
        st.session_state.active = labels.index(selected)

        model_name = st.selectbox("回答模型", list(MODELS))
        model_id = MODELS[model_name]
        show_trace = st.checkbox("显示 Agent 推理过程", value=True)
        use_demo_graph = st.checkbox("使用内置演示图谱", value=False)
        if use_demo_graph:
            st.info("演示知识库只包含抑郁症和焦虑症的图谱和文档样例，用来离线展示检索，不是完整医疗库。")

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
            st.info("未配置 DEEPSEEK_API_KEY。系统仍会调用检索工具，并把结果分段流式整理出来。")

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

    window = ensure_windows(usname)
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
    agent = CounselingAgent(knowledge, recognizer, llm, documents=build_documents(use_demo_graph))

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if show_trace and message.get("trace"):
                with st.expander("Agent 推理过程"):
                    render_trace(message["trace"])

    query = st.chat_input("想聊什么都可以，也可以问疾病、症状或用药")
    if not query:
        persist_windows(usname)
        return

    history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            answer, turn = consume_stream(agent, query, memory, show_trace)
            turn_trace = turn.trace
        except Exception:
            detail = traceback.format_exc()
            print(detail)
            answer = "处理这句话时出了问题。可以换种说法再试一次。"
            turn_trace = [{"stage": "错误", "status": "失败", "detail": "内部错误已记录在服务端日志。"}]
            st.error(answer)
            turn = None

    if turn is not None:
        memory.observe(query, answer, turn)
    history.append({"role": "assistant", "content": answer, "trace": turn_trace})
    persist_windows(usname)


if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True
        st.session_state.admin = True
        st.session_state.usname = "测试管理员"
    if st.session_state.get("logged_in", False):
        main(is_admin=st.session_state.get("admin", False), usname=st.session_state.get("usname", "用户"))
    else:
        st.error("请先登录。")
