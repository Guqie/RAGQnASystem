"""咨询 Agent：工具调用调度图谱自然语言查询、文档检索和历史记忆，再流式回答。"""

import json
from dataclasses import dataclass, field

from agent import prompts
from agent.nl2query import run_natural_query
from agent.rag import DocumentStore, format_passages
from agent.safety import CRISIS_RESPONSE, is_crisis
from agent.types import Perception

MEDICAL_CUES = (
    "治疗",
    "怎么办",
    "怎么治",
    "吃药",
    "用药",
    "什么药",
    "药品",
    "检查",
    "化验",
    "症状",
    "原因",
    "病因",
    "多久",
    "饮食",
    "忌口",
    "科室",
    "并发",
    "厂家",
    "药厂",
    "是什么",
    "介绍",
    "这是",
    "预防",
    "能好",
)
RECALL_CUES = ("上次", "之前", "还记得", "最早", "以前说过", "刚才说的")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_graph",
            "description": "把自然语言问题编译成图谱查询语句并执行，返回查询语句和结构化结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "补全指代后的自然语言检索问题"},
                    "entities": {"type": "array", "items": {"type": "string"}, "description": "已经确定的疾病名"},
                    "symptoms": {"type": "array", "items": {"type": "string"}, "description": "症状"},
                    "drug": {"type": "string", "description": "药品名"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "在疾病文档库中检索简介、病因等原文片段。",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "检索当前窗口之外的更早对话。",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
]


@dataclass
class AgentTurn:
    final_text: str | None
    messages: list | None
    trace: list
    knowledge: str
    plan: dict
    perception: Perception
    resolved_diseases: list = field(default_factory=list)
    symptoms: list = field(default_factory=list)
    intents: list = field(default_factory=list)
    drug: str | None = None
    crisis: bool = False

    @property
    def resolved_disease(self) -> str | None:
        return self.resolved_diseases[0] if self.resolved_diseases else None


def _note(stage: str, status: str, detail: str) -> dict:
    return {"type": "trace", "item": {"stage": stage, "status": status, "detail": detail}}


def iter_chunks(text: str, size: int = 24):
    buffer = ""
    for char in text or "":
        buffer += char
        if len(buffer) >= size or char in "。！？\n":
            yield buffer
            buffer = ""
    if buffer:
        yield buffer


def continues_medical(query: str) -> bool:
    return any(word in query for word in MEDICAL_CUES)


class CounselingAgent:
    def __init__(self, knowledge, recognizer, llm=None, documents=None):
        self.knowledge = knowledge
        self.recognizer = recognizer
        self.llm = llm
        self.documents = documents if documents is not None else DocumentStore()

    def step(self, query: str, memory=None) -> AgentTurn:
        chunks = []
        trace = []
        done = {}
        for event in self.stream(query, memory):
            if event["type"] == "token":
                chunks.append(event["text"])
            elif event["type"] == "trace":
                trace.append(event["item"])
            elif event["type"] == "done":
                done = event
        return AgentTurn(
            final_text="".join(chunks),
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

    def stream(self, query: str, memory=None):
        from agent.memory import SessionMemory

        memory = memory or SessionMemory()
        query = (query or "").strip()
        plan = {"source": "local", "tools": [], "use_memory": False, "reasoning": "", "statements": ""}
        if is_crisis(query):
            yield _note("安全检查", "危机表述", "停止检索和用药讨论，返回危机支持与援助热线。")
            yield _note("回答", "固定话术", "危机回复按句流式输出，不经过生成模型。")
            for piece in iter_chunks(CRISIS_RESPONSE):
                yield {"type": "token", "text": piece}
            yield self._done("", plan, Perception.empty(), [], [], None, True, memory)
            return

        yield _note("安全检查", "通过", "没有发现需要立即转介的危机表述。")
        try:
            perception = self.recognizer.recognize(query)
        except Exception as exc:
            perception = Perception.empty()
            yield _note("实体识别", "失败", f"识别器出错：{exc}")
        else:
            yield _note("实体识别", perception.backend, _format_perception(perception))

        follow = self._followup(query, perception, memory)
        if follow:
            names = "、".join(memory.focus_diseases)
            plan["use_memory"] = True
            plan["reasoning"] = f"沿用上一轮提到的疾病：{names}。"
            yield _note("记忆", "沿用上文", plan["reasoning"])
        elif memory.focus_diseases or memory.summary_lines:
            yield _note("记忆", "已载入", memory.prompt_block()[:500])

        state = {
            "knowledge_parts": [],
            "resolved": [],
            "symptoms": list(perception.symptoms),
            "drug": perception.drug,
            "statements": [],
        }
        if self._model_can_call_tools():
            yielded = yield from self._model_loop(query, memory, perception, follow, plan, state)
            if yielded:
                return
            yield _note("工具调度", "回退", "模型没有给出可执行的工具调用，改由本地按自然语言查询。")

        calls = self._local_calls(query, perception, memory, follow)
        plan["tools"] = [name for name, _ in calls]
        plan["source"] = plan.get("source") or "local"
        if not calls:
            plan["reasoning"] = plan["reasoning"] or "这轮没有医疗检索需求，直接咨询。"
            yield _note("工具调用", "未调用", plan["reasoning"])
            knowledge = prompts.COUNSEL_ONLY_KNOWLEDGE
            yield from self._speak(query, memory, knowledge, plan, perception, state)
            return

        for name, arguments in calls:
            yield {"type": "status", "text": _status_text(name)}
            detail, hit = self._invoke(name, arguments, memory)
            self._absorb(name, detail, hit, state, plan)
            yield _note(detail["stage"], detail["status"], detail["detail"])

        knowledge = self._join_knowledge(state, unavailable=not self.knowledge.available)
        plan["statements"] = "\n".join(state["statements"])
        yield from self._speak(query, memory, knowledge, plan, perception, state)

    def _model_can_call_tools(self) -> bool:
        return self.llm is not None and hasattr(self.llm, "complete_with_tools")

    def _model_loop(self, query, memory, perception, follow, plan, state):
        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT + prompts.TOOL_GUIDE},
            {
                "role": "user",
                "content": f"对话记忆：\n{memory.prompt_block()}\n\n用户当前的话：\n{query}",
            },
        ]
        try:
            first = self.llm.complete_with_tools(messages, TOOLS)
        except Exception as exc:
            yield _note("工具调度", "失败", f"工具调用接口不可用：{exc}")
            return False
        plan["source"] = "llm"
        if not first.tool_calls:
            if first.content:
                plan["reasoning"] = "模型没有调用检索工具，直接回答。"
                yield _note("工具调用", "未调用", plan["reasoning"])
                for piece in iter_chunks(first.content):
                    yield {"type": "token", "text": piece}
                yield self._done(prompts.COUNSEL_ONLY_KNOWLEDGE, plan, perception, state["resolved"], state["symptoms"], state["drug"], False, memory)
                return True
            return False

        notes = []
        for call in first.tool_calls[:3]:
            arguments = _load_arguments(call.arguments)
            if follow and call.name == "query_knowledge_graph" and not arguments.get("entities"):
                arguments.setdefault("question", query)
            yield {"type": "status", "text": _status_text(call.name)}
            detail, hit = self._invoke(call.name, arguments, memory)
            self._absorb(call.name, detail, hit, state, plan)
            plan["tools"].append(call.name)
            notes.append(f"{call.name}\n{detail['detail'][:1200]}")
            yield _note(detail["stage"], detail["status"], detail["detail"])
        plan["statements"] = "\n".join(state["statements"])
        plan["reasoning"] = plan["reasoning"] or "模型通过工具调用完成检索。"
        knowledge = self._join_knowledge(state, unavailable=not self.knowledge.available and not state["knowledge_parts"])
        if notes and not state["knowledge_parts"]:
            knowledge = knowledge or "\n\n".join(notes)
        yield from self._speak(query, memory, knowledge, plan, perception, state)
        return True

    def _local_calls(self, query, perception, memory, follow):
        if not self._wants_retrieval(query, perception, memory, follow):
            return []
        question = query
        entities = perception.focus_names()
        symptoms = list(perception.symptoms)
        drug = perception.drug
        if follow:
            question = f"关于{'、'.join(memory.focus_diseases)}：{query}"
            if not entities:
                entities = list(memory.focus_diseases)
            if not drug:
                drug = memory.last_drug
        arguments = {"question": question, "entities": entities, "symptoms": symptoms, "drug": drug}
        calls = [
            ("query_knowledge_graph", arguments),
            ("search_documents", {"question": question}),
        ]
        if any(word in query for word in RECALL_CUES) and len(getattr(memory, "history", []) or []) > 6:
            calls.append(("recall_memory", {"question": query}))
        return calls

    def _wants_retrieval(self, query, perception, memory, follow) -> bool:
        if perception.focus_names() or perception.symptoms or perception.drug:
            return True
        return bool(follow)

    def _followup(self, query, perception, memory) -> bool:
        if perception.focus_names() or not memory.focus_diseases:
            return False
        return continues_medical(query)

    def _invoke(self, name: str, arguments: dict, memory):
        question = arguments.get("question") or ""
        if name == "query_knowledge_graph":
            perceived = self.recognizer.recognize(question) if question else Perception.empty()
            entities = _merge(arguments.get("entities"), perceived.focus_names())
            symptoms = _merge(arguments.get("symptoms"), perceived.symptoms)
            drug = arguments.get("drug") or perceived.drug
            if not entities and not symptoms and memory.focus_diseases and continues_medical(question):
                entities = list(memory.focus_diseases)
            result = run_natural_query(question, entities, symptoms, drug, self.knowledge)
            stage = "工具 · 症状推断" if result.inferred else "工具 · 图谱查询"
            if result.unavailable:
                status = "跳过"
            elif result.misses and not result.hits:
                status = "未命中"
            else:
                status = "完成"
            detail = _graph_detail(result)
            hit = result.hits if result.hits else ""
            return {"stage": stage, "status": status, "detail": detail, "result": result}, hit
        if name == "search_documents":
            passages = self.documents.search(_document_query(question, arguments.get("symptoms")), k=3) if question else []
            text = format_passages(passages)
            status = "完成" if text else "无结果"
            detail = text or "文档库没有命中片段。"
            return {"stage": "工具 · 文档检索", "status": status, "detail": f"自然语言问题：{question}\n{detail}"}, text
        if name == "recall_memory":
            text = memory.recall(question)
            return {"stage": "工具 · 历史记忆", "status": "完成", "detail": text}, text
        return {"stage": "工具调用", "status": "未知工具", "detail": name}, ""

    def _absorb(self, name, detail, hit, state, plan):
        if name == "query_knowledge_graph":
            result = detail.get("result")
            if result is not None:
                state["statements"].extend(item.format() for item in result.statements)
                state["resolved"].extend(name for name in result.diseases if name not in state["resolved"])
                plan["statements"] = "\n".join(state["statements"])
            if hit:
                state["knowledge_parts"].append("【图谱】\n" + hit)
            elif result is not None and result.unavailable:
                state["knowledge_parts"].append(prompts.UNAVAILABLE_KNOWLEDGE)
        elif hit and name == "search_documents":
            state["knowledge_parts"].append(hit)
        elif hit and name == "recall_memory":
            state["knowledge_parts"].append("【历史】\n" + hit)

    def _join_knowledge(self, state, unavailable: bool) -> str:
        parts = [part for part in state["knowledge_parts"] if part and part != prompts.UNAVAILABLE_KNOWLEDGE]
        if parts:
            return "\n\n".join(parts)
        if unavailable or prompts.UNAVAILABLE_KNOWLEDGE in state["knowledge_parts"]:
            return prompts.UNAVAILABLE_KNOWLEDGE
        if state["statements"]:
            return prompts.EMPTY_KNOWLEDGE
        return prompts.COUNSEL_ONLY_KNOWLEDGE

    def _speak(self, query, memory, knowledge, plan, perception, state):
        messages = prompts.build_messages(query, memory, knowledge)
        produced = False
        if self.llm is not None and hasattr(self.llm, "stream"):
            yield {"type": "status", "text": "正在流式生成回答"}
            try:
                for token in self.llm.stream(messages):
                    produced = True
                    yield {"type": "token", "text": token}
            except Exception as exc:
                yield _note("回答", "流式失败", str(exc))
        if not produced:
            yield {"type": "status", "text": "正在流式输出整理后的回复"}
            text = prompts.offline_answer(query, memory, knowledge)
            for piece in iter_chunks(text):
                yield {"type": "token", "text": piece}
        yield _note("回答", "流式输出", "回答按片段送到界面。")
        yield self._done(
            knowledge,
            plan,
            perception,
            state["resolved"],
            state["symptoms"],
            state["drug"],
            False,
            memory,
            messages,
        )

    def _done(self, knowledge, plan, perception, resolved, symptoms, drug, crisis, memory, messages=None):
        return {
            "type": "done",
            "knowledge": knowledge,
            "plan": plan,
            "perception": perception,
            "resolved_diseases": resolved,
            "symptoms": symptoms,
            "drug": drug,
            "crisis": crisis,
            "intents": [line for line in (plan.get("statements") or "").splitlines() if line],
            "messages": messages,
        }


def _load_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"question": str(raw or "")}
    return data if isinstance(data, dict) else {}


def _merge(primary, extra) -> list:
    values = []
    for source in (primary or []), (extra or []):
        for item in source:
            if item and item not in values:
                values.append(item)
    return values


def _graph_detail(result) -> str:
    lines = [f"自然语言问题：{result.question}", "查询语句：", result.statements_text]
    if result.unavailable:
        lines.append("知识图谱未连接，查询语句已生成但没有执行。")
    elif result.hits:
        lines.append("图谱返回：")
        lines.append(result.hits)
    if result.misses:
        lines.append("未命中：" + "、".join(result.misses))
    return "\n".join(lines)


def _document_query(question: str, symptoms) -> str:
    """把已经选出的图谱字段补进文档检索，让「吃药」能对上「药品」片段。"""
    from agent.nl2query import select_fields

    hints = []
    for _, field, target in select_fields(question, bool(symptoms)):
        for hint in (field, target):
            if hint and hint not in hints:
                hints.append(hint)
        if "药" in field and "药品" not in hints:
            hints.append("药品")
    return (question + " " + " ".join(hints)).strip()


def _status_text(name: str) -> str:
    return {
        "query_knowledge_graph": "正在把自然语言编译成图谱查询",
        "search_documents": "正在检索疾病文档库",
        "recall_memory": "正在查找更早的对话",
    }.get(name, "正在调用工具")


def _format_perception(perception: Perception) -> str:
    if not perception.by_type and not perception.focus_names() and not perception.symptoms and not perception.drug:
        return "没有识别到词典中的医疗实体。"
    lines = []
    if perception.by_type:
        for etype, names in perception.by_type.items():
            lines.append(f"{etype}：{'、'.join(names)}")
        return "\n".join(lines)
    if perception.focus_names():
        lines.append("疾病：" + "、".join(perception.focus_names()))
    if perception.symptoms:
        lines.append("疾病症状：" + "、".join(perception.symptoms))
    if perception.drug:
        lines.append(f"药品：{perception.drug}")
    return "\n".join(lines)
