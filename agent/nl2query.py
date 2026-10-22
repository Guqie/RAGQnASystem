"""把自然语言编译成毕设里的图谱查询语句，再执行并返回结果。

语句格式和 nl2cypher 语料一致：
- `1 实体 属性`：读疾病节点上的属性
- `2 实体 关系 目标类型`：沿关系取邻居；目标类型写成「疾病」时表示从症状反查疾病
"""

from dataclasses import dataclass

from agent.knowledge import render_candidates
from agent.schema import ATTRIBUTE_INTENTS, RELATION_INTENTS

FIELD_TO_INTENT = {
    "疾病简介": "查询疾病简介",
    "疾病病因": "查询疾病病因",
    "预防措施": "查询疾病预防措施",
    "治疗周期": "查询疾病治疗周期",
    "治愈概率": "查询治愈概率",
    "疾病易感人群": "查询疾病易感人群",
    "疾病使用药品": "查询疾病常用药品",
    "治疗的方法": "查询疾病的治疗方法",
    "疾病宜吃食物": "查询疾病宜吃食物",
    "疾病忌吃食物": "查询疾病忌吃食物",
    "疾病所需检查": "查询疾病所需检查",
    "疾病所属科目": "查询疾病所属科室",
    "疾病的症状": "查询疾病的症状",
    "疾病并发疾病": "查询疾病的并发疾病",
}


@dataclass(frozen=True)
class Statement:
    kind: int
    entity: str
    field: str
    target: str = ""

    def format(self) -> str:
        if self.kind == 1:
            return f"1 {self.entity} {self.field}"
        return f"2 {self.entity} {self.field} {self.target}".strip()


@dataclass
class NLResult:
    question: str
    statements: list
    hits: str
    diseases: list
    inferred: bool = False
    misses: list | None = None
    unavailable: bool = False

    @property
    def statements_text(self) -> str:
        if not self.statements:
            return "（没有生成查询语句）"
        return "\n".join(item.format() for item in self.statements)


def _has_any(question: str, words: tuple) -> bool:
    return any(word in question for word in words)


def select_fields(question: str, has_symptoms: bool) -> list:
    """返回 (kind, field, target) ，属性在前，关系在后，最多 3 项。"""
    fields = []

    def add(kind: int, field: str, target: str = "") -> None:
        item = (kind, field, target)
        if item not in fields and len(fields) < 3:
            fields.append(item)

    if _has_any(question, ("是什么", "什么病", "介绍", "这是")):
        add(1, "疾病简介")
    if _has_any(question, ("原因", "为什么", "病因", "怎么得")):
        add(1, "疾病病因")
    if _has_any(question, ("预防", "怎么避免", "注意什么")):
        add(1, "预防措施")
    if _has_any(question, ("多久", "多长时间", "治疗周期")):
        add(1, "治疗周期")
    if _has_any(question, ("能好", "治愈", "预后", "好得了")):
        add(1, "治愈概率")
    if _has_any(question, ("哪些人", "易感", "谁容易")):
        add(1, "疾病易感人群")
    if _has_any(question, ("吃什么药", "吃药", "用药", "什么药", "药品")):
        add(2, "疾病使用药品", "药品")
    elif _has_any(question, ("吃什么", "宜吃", "饮食", "食谱")):
        add(2, "疾病宜吃食物", "食物")
    if _has_any(question, ("不能吃", "忌口", "忌吃")):
        add(2, "疾病忌吃食物", "食物")
    if _has_any(question, ("检查", "化验", "确诊")):
        add(2, "疾病所需检查", "检查项目")
    if _has_any(question, ("科室", "挂什么", "看哪科", "挂号")):
        add(2, "疾病所属科目", "科目")
    if _has_any(question, ("症状", "表现", "是不是")):
        add(2, "疾病的症状", "疾病症状")
    if _has_any(question, ("并发", "并发症", "还会得")):
        add(2, "疾病并发疾病", "疾病")
    if _has_any(question, ("治疗", "怎么办", "怎么治", "如何处理")):
        add(2, "治疗的方法", "治疗方法")

    if not fields:
        add(1, "疾病简介")
        if has_symptoms:
            add(2, "疾病的症状", "疾病症状")
    elif has_symptoms and (2, "疾病的症状", "疾病症状") not in fields:
        add(2, "疾病的症状", "疾病症状")
    return fields


def compile_statements(question: str, entities: list, symptoms: list, drug: str | None) -> list:
    if drug and _has_any(question, ("生产商", "厂家", "药厂", "哪家生产", "哪家")):
        return [Statement(2, drug, "生产", "药品商")]

    fields = select_fields(question, bool(symptoms))
    statements = []
    for name in entities[:2]:
        for kind, field, target in fields:
            statements.append(Statement(kind, name, field, target))
            if len(statements) >= 4:
                return statements
    if statements:
        return statements
    for symptom in symptoms[:3]:
        statements.append(Statement(2, symptom, "疾病的症状", "疾病"))
    return statements


def _bind(entity: str, fields: list) -> list:
    bound = []
    for kind, field, target in fields:
        bound.append(Statement(kind, entity, field, target))
        if len(bound) >= 3:
            break
    return bound


def run_natural_query(question: str, entities: list, symptoms: list, drug: str | None, knowledge) -> NLResult:
    statements = compile_statements(question, entities, symptoms, drug)
    if not getattr(knowledge, "available", False):
        return NLResult(question, statements, "", [], unavailable=True, misses=[])

    hits = []
    diseases = []
    misses = []
    inferred = False
    reverse = [item.entity for item in statements if item.field == "疾病的症状" and item.target == "疾病"]
    forward = [
        item for item in statements
        if item.field != "生产" and not (item.field == "疾病的症状" and item.target == "疾病")
    ]
    producers = [item.entity for item in statements if item.field == "生产"]

    if reverse:
        rows = knowledge.diseases_by_symptoms(reverse)
        hits.append(render_candidates(rows))
        inferred = True
        if rows and not any(item.kind == 1 or item.target not in {"", "疾病"} for item in forward):
            top = rows[0]["name"]
            extra = _bind(top, select_fields(question, True))
            statements = list(statements) + extra
            forward.extend(extra)

    grouped = {}
    for item in forward:
        if item.field == "生产":
            continue
        intent = FIELD_TO_INTENT.get(item.field)
        if not intent:
            continue
        bucket = grouped.setdefault(item.entity, [])
        if intent not in bucket:
            bucket.append(intent)

    for entity, intents in grouped.items():
        resolved = knowledge.resolve_disease(entity)
        if not resolved:
            misses.append(entity)
            continue
        profile = knowledge.query(resolved, intents)
        if not profile.text or profile.text.startswith("图谱中没有") or "未连接" in profile.text:
            misses.append(entity)
            continue
        hits.append(profile.text)
        name = profile.disease or resolved
        if name not in diseases:
            diseases.append(name)

    for name in producers:
        found = knowledge.drug_producers(name)
        if found:
            hits.append(f"药品：{name}\n生产商：{'、'.join(found)}")
        else:
            misses.append(name)

    return NLResult(
        question=question,
        statements=statements,
        hits="\n\n".join(part for part in hits if part),
        diseases=diseases,
        inferred=inferred,
        misses=misses,
    )


def intent_labels() -> dict:
    """给测试确认属性意图表仍然覆盖建图字段。"""
    return {"attributes": dict(ATTRIBUTE_INTENTS), "relations": dict(RELATION_INTENTS)}
