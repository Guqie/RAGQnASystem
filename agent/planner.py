"""规划器：优先让大模型选择工具，解析失败时退回关键词规则。"""

import json
import re

from agent.memory import SessionMemory
from agent.schema import INTENT_KEYWORDS, PREDEFINED_INTENTS
from agent.types import Perception

ALLOWED_TOOLS = {"query_knowledge", "infer_diseases", "query_drug", "counsel_only"}
FOLLOWUP_CUES = (
    "那",
    "这个",
    "这种情况",
    "这个病",
    "它",
    "还",
    "呢",
    "然后",
    "继续",
    "用药",
    "吃药",
    "治疗",
    "检查",
    "饮食",
)

PLAN_SYSTEM = """你是心理咨询系统里的规划器，不直接回答用户。
你只能从给定工具里选择，并把计划写成一个 JSON 对象。

工具：
- infer_diseases：参数 symptoms（字符串数组）。用户只描述了症状、没有明确疾病名时使用。
- query_knowledge：参数 disease（疾病名，或 "$top" 表示使用上一步推断出的疾病）、intents（最多 3 个，必须来自预定义列表）。
- query_drug：参数 drug。只有用户在问某种药品的生产厂家时使用。
- counsel_only：问候、道谢、单纯倾诉，不需要医学检索时使用。

要求：
- 最多 3 个动作，按执行顺序放进 actions。
- 如果需要先推断疾病再查询，第一个动作是 infer_diseases，第二个是 query_knowledge，disease 填 "$top"。
- 用户在追问，且本轮没有新的疾病名时，disease 使用记忆中的疾病，并把 use_memory 设为 true。
- 不要发明预定义列表之外的意图。
- 只输出 JSON，不要解释。

JSON 格式：
{"reasoning":"一句话","use_memory":false,"intents":["查询疾病简介"],"actions":[{"tool":"query_knowledge","disease":"抑郁症","intents":["查询疾病简介"]}]}
"""


def select_intents(query: str) -> list:
    scored = []
    for index, intent in enumerate(PREDEFINED_INTENTS):
        words = INTENT_KEYWORDS.get(intent, [])
        score = sum(1 for word in words if word and word in query)
        if score:
            scored.append((-score, index, intent))
    scored.sort()
    return [intent for _, _, intent in scored[:3]]


def is_followup(query: str, has_disease: bool, memory: SessionMemory) -> bool:
    if has_disease or not memory.last_disease:
        return False
    compact = "".join(query.split())
    if len(compact) <= 16:
        return True
    return any(cue in query for cue in FOLLOWUP_CUES)


def rule_plan(query: str, perception: Perception, memory: SessionMemory) -> dict:
    intents = select_intents(query)
    follow = is_followup(query, bool(perception.focus_names()), memory)
    targets = perception.focus_names()
    use_memory = False
    if not targets and follow and intents:
        targets = list(memory.focus_diseases[:2])
        use_memory = bool(targets)
    drug = perception.drug or (memory.last_drug if follow else None)

    actions = []
    if "查询药品的生产商" in intents and drug and not perception.focus_names():
        actions.append({"tool": "query_drug", "drug": drug})
        intents = ["查询药品的生产商"]
        reasoning = f"用户在问药品「{drug}」的生产商。"
    elif perception.symptoms and not targets:
        symptoms = perception.symptoms[:6]
        lookup_intents = intents or ["查询疾病简介", "查询疾病的症状"]
        lookup_intents = lookup_intents[:3]
        actions.append({"tool": "infer_diseases", "symptoms": symptoms})
        actions.append(
            {"tool": "query_knowledge", "disease": "$top", "intents": lookup_intents}
        )
        intents = lookup_intents
        reasoning = (
            f"问句里有症状（{'、'.join(symptoms)}），没有明确疾病名。"
            "先按图谱共现推断疾病，再查询相关资料。"
        )
    elif targets:
        lookup_intents = (intents or ["查询疾病简介", "查询疾病的症状"])[:3]
        if perception.symptoms and "查询疾病的症状" not in lookup_intents and len(lookup_intents) < 3:
            lookup_intents.append("查询疾病的症状")
        for name in targets[:2]:
            actions.append(
                {"tool": "query_knowledge", "disease": name, "intents": lookup_intents}
            )
        intents = lookup_intents
        source = "沿用上一轮提到的疾病" if use_memory else "使用本轮识别到的疾病"
        reasoning = f"{source}：{'、'.join(targets[:2])}。准备查询：{'、'.join(intents)}。"
    else:
        actions.append({"tool": "counsel_only"})
        intents = []
        reasoning = "没有识别到需要检索的疾病、症状或药品，本轮直接做心理咨询。"

    return {
        "reasoning": reasoning,
        "source": "rule",
        "use_memory": use_memory,
        "intents": intents,
        "actions": actions,
    }


def extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    candidates = []
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    start = text.find("{")
    scanned = 0
    while start != -1 and scanned < 6:
        depth = 0
        for index, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
        scanned += 1
        start = text.find("{", start + 1)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _clean_intent(value) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().replace("\u200b", "")
    if cleaned in PREDEFINED_INTENTS:
        return cleaned
    return None


def normalize_plan(obj: dict | None) -> dict | None:
    if not isinstance(obj, dict):
        return None
    intents = []
    for item in obj.get("intents") or []:
        cleaned = _clean_intent(item)
        if cleaned and cleaned not in intents:
            intents.append(cleaned)
        if len(intents) >= 3:
            break

    actions = []
    for raw in obj.get("actions") or []:
        if not isinstance(raw, dict):
            continue
        tool = raw.get("tool")
        if tool not in ALLOWED_TOOLS:
            continue
        action = {"tool": tool}
        if tool == "query_knowledge":
            disease = raw.get("disease") or "$top"
            action["disease"] = str(disease).strip() or "$top"
            chosen = []
            for item in raw.get("intents") or intents:
                cleaned = _clean_intent(item)
                if cleaned and cleaned not in chosen:
                    chosen.append(cleaned)
                if len(chosen) >= 3:
                    break
            action["intents"] = chosen or intents or ["查询疾病简介"]
        elif tool == "infer_diseases":
            symptoms = []
            for item in raw.get("symptoms") or []:
                if isinstance(item, str) and item.strip() and item.strip() not in symptoms:
                    symptoms.append(item.strip())
                if len(symptoms) >= 6:
                    break
            action["symptoms"] = symptoms
        elif tool == "query_drug":
            drug = raw.get("drug")
            action["drug"] = str(drug).strip() if isinstance(drug, str) else ""
        actions.append(action)
        if len(actions) >= 3:
            break
    if not actions:
        return None
    reasoning = str(obj.get("reasoning") or "").strip()
    return {
        "reasoning": reasoning[:300] or "模型给出了工具计划。",
        "source": "llm",
        "use_memory": bool(obj.get("use_memory")),
        "intents": intents,
        "actions": actions,
    }


def _plan_user(query: str, perception: Perception, memory: SessionMemory) -> str:
    diseases = "、".join(perception.focus_names()) or "无"
    symptoms = "、".join(perception.symptoms) or "无"
    drug = perception.drug or "无"
    intent_lines = "\n".join(f"- {intent}" for intent in PREDEFINED_INTENTS)
    return f"""预定义意图：
{intent_lines}

对话记忆：
{memory.prompt_block()}

本轮实体：
- 疾病：{diseases}
- 症状：{symptoms}
- 药品：{drug}

用户：{query}
"""


def llm_plan(llm, query: str, perception: Perception, memory: SessionMemory):
    messages = [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": _plan_user(query, perception, memory)},
    ]
    try:
        raw = llm.complete(messages, temperature=0.1, max_tokens=500)
    except Exception as exc:
        return None, f"规划模型调用失败：{exc}"
    plan = normalize_plan(extract_json_object(raw))
    return plan, raw or ""
