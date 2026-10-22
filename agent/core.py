"""咨询 Agent 的主循环：安全检查 → 感知 → 规划 → 工具 → 回答。"""

from dataclasses import dataclass, field

from agent import prompts
from agent.knowledge import render_candidates
from agent.memory import SessionMemory
from agent.planner import llm_plan, rule_plan
from agent.safety import CRISIS_RESPONSE, is_crisis
from agent.types import Perception


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


def _note(trace: list, stage: str, status: str, detail: str) -> None:
    trace.append({"stage": stage, "status": status, "detail": detail})


class CounselingAgent:
    def __init__(self, knowledge, recognizer, llm=None):
        self.knowledge = knowledge
        self.recognizer = recognizer
        self.llm = llm

    def step(self, query: str, memory: SessionMemory | None = None) -> AgentTurn:
        memory = memory or SessionMemory()
        query = (query or "").strip()
        trace = []

        if is_crisis(query):
            _note(trace, "安全检查", "危机表述", "停止检索和用药讨论，返回危机支持与援助热线。")
            _note(trace, "回答", "固定话术", "危机回复不经过生成模型，避免被改写成用药建议。")
            return AgentTurn(
                final_text=CRISIS_RESPONSE,
                messages=None,
                trace=trace,
                knowledge="",
                plan={"reasoning": "危机优先", "source": "safety", "actions": [], "intents": []},
                perception=Perception.empty(),
                crisis=True,
            )

        _note(trace, "安全检查", "通过", "没有发现需要立即转介的危机表述。")

        try:
            perception = self.recognizer.recognize(query)
        except Exception as exc:
            perception = Perception.empty()
            _note(trace, "实体识别", "失败", f"识别器出错，改为不依赖实体继续：{exc}")
        else:
            _note(trace, "实体识别", perception.backend, _format_perception(perception))

        if memory.focus_diseases or memory.last_symptoms:
            remembered = "、".join(memory.focus_diseases) or "无"
            _note(trace, "记忆", "已载入", f"关注疾病：{remembered}")

        plan = None
        raw_plan = ""
        if self.llm is not None:
            plan, raw_plan = llm_plan(self.llm, query, perception, memory)
        if plan:
            detail = plan["reasoning"]
            if raw_plan:
                detail += "\n\n模型原文：" + raw_plan.strip()[:500]
            _note(trace, "规划", "大模型", detail)
        else:
            plan = rule_plan(query, perception, memory)
            status = "规则"
            if self.llm is not None:
                status = "规则回退"
                extra = f"模型输出无法解析，已改用规则。{raw_plan}" if raw_plan else "模型输出无法解析，已改用规则。"
                plan = dict(plan)
                plan["reasoning"] = extra + plan["reasoning"]
            _note(trace, "规划", status, plan["reasoning"])

        symptoms = list(perception.symptoms)
        drug = perception.drug
        if not self.knowledge.available and _needs_graph(plan):
            knowledge = prompts.UNAVAILABLE_KNOWLEDGE
            _note(trace, "工具", "跳过", "知识图谱未连接，计划中的检索没有执行。")
            resolved = []
        else:
            resolved, symptoms, drug, knowledge, tool_trace = self._execute(plan, perception, memory)
            trace.extend(tool_trace)

        messages = prompts.build_messages(query, memory, knowledge)
        final_text = None if self.llm is not None else prompts.offline_answer(query, memory, knowledge)
        _note(
            trace,
            "回答",
            "大模型" if self.llm is not None else "离线整理",
            "根据记忆和检索结果组织咨询回复。" if self.llm is not None else "未配置大模型，直接整理检索结果。",
        )
        return AgentTurn(
            final_text=final_text,
            messages=messages,
            trace=trace,
            knowledge=knowledge,
            plan=plan,
            perception=perception,
            resolved_diseases=resolved,
            symptoms=symptoms,
            intents=list(plan.get("intents") or []),
            drug=drug,
        )

    def _execute(self, plan: dict, perception: Perception, memory: SessionMemory):
        resolved = []
        symptoms = list(perception.symptoms)
        drug = perception.drug
        top = None
        chunks = []
        trace = []
        queried = False

        for action in plan.get("actions") or []:
            tool = action.get("tool")
            if tool == "counsel_only":
                _note(trace, "工具", "直接咨询", "本轮不查询知识图谱。")
                continue
            if tool == "infer_diseases":
                asked = action.get("symptoms") or symptoms
                symptoms = list(asked) or symptoms
                rows = self.knowledge.diseases_by_symptoms(asked)
                top = rows[0]["name"] if rows else None
                text = render_candidates(rows)
                chunks.append(text)
                _note(trace, "工具 · 症状推断", "完成" if rows else "无结果", text)
                continue
            if tool == "query_drug":
                drug_name = action.get("drug") or drug
                drug = drug_name or drug
                producers = self.knowledge.drug_producers(drug_name) if drug_name else []
                if producers:
                    text = f"药品：{drug_name}\n生产商：{'、'.join(producers)}"
                elif drug_name:
                    text = f"药品：{drug_name}\n生产商：图谱中未收录"
                else:
                    text = "没有识别到药品名称，无法查询生产商。"
                chunks.append(text)
                _note(trace, "工具 · 药品生产商", "完成", text)
                continue
            if tool != "query_knowledge":
                continue
            name = self._bind_disease(
                action.get("disease"),
                perception,
                memory,
                top,
                plan.get("use_memory"),
            )
            if not name:
                text = f"没有在图谱中定位到疾病「{action.get('disease') or '未提供'}」。"
                _note(trace, "工具 · 疾病查询", "未命中", text)
                continue
            intents = action.get("intents") or plan.get("intents") or ["查询疾病简介"]
            profile = self.knowledge.query(name, intents)
            disease_name = profile.disease or name
            if disease_name not in resolved:
                resolved.append(disease_name)
            chunks.append(profile.text)
            queried = True
            _note(trace, "工具 · 疾病查询", disease_name, profile.text)

        if top and not queried and plan.get("intents"):
            profile = self.knowledge.query(top, plan["intents"])
            if profile.disease and profile.disease not in resolved:
                resolved.append(profile.disease)
            chunks.append(profile.text)
            _note(trace, "工具 · 第二跳", profile.disease or top, "推断之后补查该疾病的资料。\n" + profile.text)

        if not chunks:
            knowledge = prompts.COUNSEL_ONLY_KNOWLEDGE if _only_counsel(plan) else prompts.EMPTY_KNOWLEDGE
        else:
            knowledge = "\n\n".join(chunks)
        return resolved, symptoms, drug, knowledge, trace

    def _bind_disease(self, raw, perception: Perception, memory: SessionMemory, top, use_memory: bool):
        token = (raw or "").strip()
        if token in {"", "$top", "$inferred_top"}:
            if top:
                return top
            if use_memory and memory.last_disease:
                return memory.last_disease
            if perception.disease:
                return self.knowledge.resolve_disease(perception.disease)
            return None
        resolved = self.knowledge.resolve_disease(token)
        if resolved:
            return resolved
        if use_memory and memory.last_disease:
            return memory.last_disease
        return None


def _needs_graph(plan: dict) -> bool:
    return any(action.get("tool") in {"query_knowledge", "infer_diseases", "query_drug"} for action in plan.get("actions") or [])


def _only_counsel(plan: dict) -> bool:
    actions = plan.get("actions") or []
    return bool(actions) and all(action.get("tool") == "counsel_only" for action in actions)


def _format_perception(perception: Perception) -> str:
    if not perception.by_type and not perception.focus_names() and not perception.symptoms:
        return "没有识别到词典中的医疗实体。"
    lines = []
    if perception.by_type:
        for etype, names in perception.by_type.items():
            lines.append(f"{etype}：{'、'.join(names)}")
    else:
        if perception.focus_names():
            lines.append("疾病：" + "、".join(perception.focus_names()))
        if perception.symptoms:
            lines.append("疾病症状：" + "、".join(perception.symptoms))
        if perception.drug:
            lines.append(f"药品：{perception.drug}")
    return "\n".join(lines)

