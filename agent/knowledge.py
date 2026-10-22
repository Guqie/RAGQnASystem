"""知识图谱工具。Neo4j 与内存图谱实现同一组查询。"""

from agent.schema import ATTR_EXPR, ATTRIBUTE_INTENTS, RELATION_INTENTS
from agent.types import Profile


def clip(value, limit: int = 360) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def render_profile(disease: str, attributes: dict, relations: dict) -> str:
    lines = [f"疾病：{disease}"]
    for key, value in attributes.items():
        lines.append(f"{key}：{clip(value) if value else '图谱中未收录'}")
    for key, items in relations.items():
        if items:
            shown = "、".join(str(item) for item in items[:12])
            lines.append(f"{key}：{shown}")
        else:
            lines.append(f"{key}：图谱中未收录")
    return "\n".join(lines)


def render_candidates(rows: list) -> str:
    if not rows:
        return "知识图谱里没有和这些症状共现的疾病。这不能说明用户没有患病，只是图谱没有对应条目。"
    parts = [f"{row['name']}（共现症状 {row['hit']} 项）" for row in rows]
    return (
        "根据症状在知识图谱中的共现，可能相关的疾病有："
        + "、".join(parts)
        + "。这是图谱里的共现统计，不是诊断。"
    )


class Neo4jKnowledge:
    def __init__(self, client):
        self.client = client

    @property
    def available(self) -> bool:
        return self.client is not None

    def _run(self, cypher: str, **params) -> list:
        result = self.client.run(cypher, **params)
        if hasattr(result, "data"):
            return result.data()
        return list(result)

    def resolve_disease(self, name: str) -> str | None:
        if not name or not self.client:
            return None
        try:
            exact = self._run(
                "MATCH (d:疾病 {名称: $name}) RETURN d.名称 AS name LIMIT 1",
                name=name,
            )
        except Exception:
            return None
        if exact and exact[0].get("name"):
            return exact[0]["name"]
        if not name.endswith("症"):
            try:
                suffixed = self._run(
                    "MATCH (d:疾病 {名称: $name}) RETURN d.名称 AS name LIMIT 1",
                    name=name + "症",
                )
            except Exception:
                suffixed = []
            if suffixed and suffixed[0].get("name"):
                return suffixed[0]["name"]
        try:
            fuzzy = self._run(
                "MATCH (d:疾病) WHERE d.名称 CONTAINS $name RETURN d.名称 AS name LIMIT 5",
                name=name,
            )
        except Exception:
            return None
        names = [row.get("name") for row in fuzzy if row.get("name")]
        if not names:
            return None
        names.sort(key=len)
        return names[0]

    def query(self, name: str, intents: list) -> Profile:
        if not self.client:
            return Profile(name, "知识图谱未连接，无法查询疾病信息。")
        attributes = {}
        relations = {}
        attr_intents = [item for item in intents if item in ATTRIBUTE_INTENTS]
        if attr_intents:
            pieces = []
            labels = []
            for intent in attr_intents:
                label = ATTRIBUTE_INTENTS[intent]
                if label in labels:
                    continue
                labels.append(label)
                pieces.append(f"{ATTR_EXPR[label]} AS `{label}`")
            cypher = (
                f"MATCH (d:疾病 {{名称: $name}}) RETURN {', '.join(pieces)} LIMIT 1"
            )
            try:
                rows = self._run(cypher, name=name)
            except Exception as exc:
                return Profile(name, f"查询「{name}」的属性时失败：{exc}")
            row = rows[0] if rows else {}
            for label in labels:
                attributes[label] = row.get(label) or ""

        for intent in intents:
            if intent not in RELATION_INTENTS:
                continue
            label, rel_types = RELATION_INTENTS[intent]
            rel = "|".join(rel_types)
            cypher = (
                f"MATCH (d:疾病 {{名称: $name}})-[:{rel}]->(n) "
                "RETURN DISTINCT n.名称 AS name LIMIT 12"
            )
            try:
                rows = self._run(cypher, name=name)
            except Exception as exc:
                relations[label] = []
                attributes.setdefault("查询错误", str(exc))
                continue
            relations[label] = [row.get("name") for row in rows if row.get("name")]

        if not attributes and not relations:
            return Profile(name, f"没有为「{name}」选择可执行的查询。")
        return Profile(name, render_profile(name, attributes, relations), attributes, relations)

    def diseases_by_symptoms(self, symptoms: list) -> list:
        cleaned = [item for item in symptoms if item]
        if not cleaned or not self.client:
            return []
        cypher = (
            "MATCH (d:疾病)-[:疾病的症状]->(s:疾病症状) "
            "WHERE s.名称 IN $symptoms "
            "WITH d, count(DISTINCT s) AS hit "
            "RETURN d.名称 AS name, hit "
            "ORDER BY hit DESC, size(d.名称) ASC "
            "LIMIT 5"
        )
        try:
            rows = self._run(cypher, symptoms=cleaned)
        except Exception:
            return []
        return [
            {"name": row.get("name"), "hit": row.get("hit", 0)}
            for row in rows
            if row.get("name")
        ]

    def drug_producers(self, drug: str) -> list:
        if not drug or not self.client:
            return []
        cypher = (
            "MATCH (p:药品商)-[:生产]->(dr:药品 {名称: $name}) "
            "RETURN DISTINCT p.名称 AS name LIMIT 5"
        )
        try:
            rows = self._run(cypher, name=drug)
        except Exception:
            return []
        return [row.get("name") for row in rows if row.get("name")]


class InMemoryKnowledge:
    """离线演示和测试用的小图谱，接口与 Neo4jKnowledge 一致。"""

    def __init__(self, records: dict, producers: dict | None = None):
        self.records = records
        self.producers = producers or {}

    @property
    def available(self) -> bool:
        return True

    def resolve_disease(self, name: str) -> str | None:
        if not name:
            return None
        if name in self.records:
            return name
        if name + "症" in self.records:
            return name + "症"
        contained = [item for item in self.records if name in item]
        if not contained:
            return None
        contained.sort(key=len)
        return contained[0]

    def query(self, name: str, intents: list) -> Profile:
        resolved = self.resolve_disease(name)
        if not resolved:
            return Profile(name or "", f"图谱中没有「{name}」。")
        record = self.records[resolved]
        attributes = {}
        relations = {}
        for intent in intents:
            if intent in ATTRIBUTE_INTENTS:
                label = ATTRIBUTE_INTENTS[intent]
                attributes[label] = record.get("attributes", {}).get(label, "")
            elif intent in RELATION_INTENTS:
                label = RELATION_INTENTS[intent][0]
                relations[label] = list(record.get("relations", {}).get(label, []))
        if not attributes and not relations:
            attributes["简介"] = record.get("attributes", {}).get("简介", "")
        return Profile(
            resolved,
            render_profile(resolved, attributes, relations),
            attributes,
            relations,
        )

    def diseases_by_symptoms(self, symptoms: list) -> list:
        asked = {item for item in symptoms if item}
        if not asked:
            return []
        scored = []
        for disease, record in self.records.items():
            owned = set(record.get("relations", {}).get("症状", []))
            hit = len(owned & asked)
            if hit:
                scored.append({"name": disease, "hit": hit})
        scored.sort(key=lambda row: (-row["hit"], len(row["name"])))
        return scored[:5]

    def drug_producers(self, drug: str) -> list:
        if not drug:
            return []
        return list(self.producers.get(drug, []))
