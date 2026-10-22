import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TurnRecord:
    query: str
    answer_brief: str
    diseases: list = field(default_factory=list)
    intents: list = field(default_factory=list)


def _record_from_dict(data: dict) -> TurnRecord:
    return TurnRecord(
        query=data.get("query") or "",
        answer_brief=data.get("answer_brief") or "",
        diseases=list(data.get("diseases") or []),
        intents=list(data.get("intents") or []),
    )


@dataclass
class SessionMemory:
    """长上下文：最近几轮保留原文，更早的轮次压成摘要，全部历史可检索、可落盘。"""

    turns: list = field(default_factory=list)
    history: list = field(default_factory=list)
    summary_lines: list = field(default_factory=list)
    focus_diseases: list = field(default_factory=list)
    last_symptoms: list = field(default_factory=list)
    last_drug: str | None = None
    crisis_turns_left: int = 0

    @property
    def last_disease(self) -> str | None:
        return self.focus_diseases[0] if self.focus_diseases else None

    def _recent(self) -> list:
        if self.history:
            return self.history[-6:]
        return self.turns[-6:]

    def prompt_block(self) -> str:
        recent = self._recent()
        if not recent and not self.focus_diseases and not self.summary_lines:
            return "（这是对话的开始，还没有历史）"
        lines = []
        if self.summary_lines:
            lines.append("更早对话的摘要（长上下文压缩）：")
            lines.extend(self.summary_lines[-12:])
        if self.focus_diseases:
            lines.append("当前关注的疾病：" + "、".join(self.focus_diseases))
        if self.last_symptoms:
            lines.append("用户提到的症状：" + "、".join(self.last_symptoms))
        if self.last_drug:
            lines.append(f"用户提到的药品：{self.last_drug}")
        if self.crisis_turns_left:
            lines.append("近期出现过危机表述，回复时要先确认人身安全。")
        for index, turn in enumerate(recent, 1):
            lines.append(f"{index}. 用户：{turn.query}")
            if turn.answer_brief:
                lines.append(f"   助手摘要：{turn.answer_brief}")
        text = "\n".join(lines)
        if len(text) > 4500:
            text = text[-4500:]
        return text

    def recall(self, question: str, limit: int = 3) -> str:
        older = self.history[:-6] if len(self.history) > 6 else []
        if not older and not self.summary_lines:
            return "近期对话已经在上下文窗口里，没有更早的历史。"
        needles = set(tokenize_overlap(question))
        ranked = []
        for record in older:
            hay = set(tokenize_overlap(record.query + record.answer_brief))
            score = len(needles & hay)
            if score:
                ranked.append((score, record))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            preview = "；".join(self.summary_lines[-3:]) or "没有更早的原话。"
            return "没有按字面命中更早的原话。摘要：" + preview
        lines = []
        for _, record in ranked[:limit]:
            lines.append(f"用户当时说：{record.query}")
            if record.answer_brief:
                lines.append(f"回复要点：{record.answer_brief}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        source = self.history or self.turns
        return {
            "history": [
                {
                    "query": item.query,
                    "answer_brief": item.answer_brief,
                    "diseases": list(item.diseases),
                    "intents": list(item.intents),
                }
                for item in source
            ],
            "summary_lines": list(self.summary_lines),
            "focus_diseases": list(self.focus_diseases),
            "last_symptoms": list(self.last_symptoms),
            "last_drug": self.last_drug,
            "crisis_turns_left": self.crisis_turns_left,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMemory":
        memory = cls(
            history=[_record_from_dict(item) for item in data.get("history") or []],
            focus_diseases=list(data.get("focus_diseases") or []),
            last_symptoms=list(data.get("last_symptoms") or []),
            last_drug=data.get("last_drug"),
            crisis_turns_left=int(data.get("crisis_turns_left") or 0),
        )
        memory._refresh_context()
        return memory

    def observe(self, query: str, answer: str, turn) -> None:
        if getattr(turn, "crisis", False):
            self.crisis_turns_left = 2
        elif self.crisis_turns_left:
            self.crisis_turns_left -= 1

        diseases = list(getattr(turn, "resolved_diseases", []) or [])
        if diseases and not getattr(turn, "crisis", False):
            self.focus_diseases = diseases[:2]
        symptoms = list(getattr(turn, "symptoms", []) or [])
        if symptoms and not getattr(turn, "crisis", False):
            self.last_symptoms = symptoms[:6]
        drug = getattr(turn, "drug", None)
        if drug and not getattr(turn, "crisis", False):
            self.last_drug = drug

        brief = " ".join((answer or "").split())
        if len(brief) > 120:
            brief = brief[:120] + "…"
        record = TurnRecord(
            query=query,
            answer_brief=brief,
            diseases=list(self.focus_diseases),
            intents=list(getattr(turn, "intents", []) or []),
        )
        self.history.append(record)
        if len(self.history) > 30:
            self.history = self.history[-30:]
        self._refresh_context()

    def _refresh_context(self) -> None:
        if len(self.history) > 6:
            self.summary_lines = [self._summarize(item) for item in self.history[:-6]][-12:]
            self.turns = self.history[-6:]
        else:
            self.summary_lines = []
            self.turns = list(self.history)

    @staticmethod
    def _summarize(record: TurnRecord) -> str:
        focus = "、".join(record.diseases) if record.diseases else "情绪支持"
        return f"用户问过「{record.query[:60]}」，当时关注{focus}。"


def memory_path(root, username: str) -> Path:
    safe = re.sub(r"[^\w\u4e00-\u9fff-]", "_", username or "user")
    safe = safe.strip("._") or "user"
    return Path(root) / "tmp_data" / "memory" / f"{safe}.json"


def save_session(path, windows: list, active: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "active": active,
        "windows": [
            {
                "messages": list(window.get("messages") or []),
                "memory": window["memory"].to_dict(),
            }
            for window in windows
        ],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_session(path):
    target = Path(path)
    if not target.exists():
        return [{"messages": [], "memory": SessionMemory()}], 0
    data = json.loads(target.read_text(encoding="utf-8"))
    windows = []
    for item in data.get("windows") or []:
        windows.append(
            {
                "messages": list(item.get("messages") or []),
                "memory": SessionMemory.from_dict(item.get("memory") or {}),
            }
        )
    if not windows:
        windows = [{"messages": [], "memory": SessionMemory()}]
    active = int(data.get("active") or 0)
    active = min(max(active, 0), len(windows) - 1)
    return windows, active


def tokenize_overlap(text: str) -> list:
    compact = "".join(str(text).split())
    if len(compact) < 2:
        return [compact] if compact else []
    return [compact[i : i + 2] for i in range(len(compact) - 1)]
