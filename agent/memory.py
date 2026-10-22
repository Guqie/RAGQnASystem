from dataclasses import dataclass, field


@dataclass
class TurnRecord:
    query: str
    answer_brief: str
    diseases: list = field(default_factory=list)
    intents: list = field(default_factory=list)


@dataclass
class SessionMemory:
    """一个对话窗口里的短期记忆。只保留最近几轮，供规划和回答使用。"""

    turns: list = field(default_factory=list)
    focus_diseases: list = field(default_factory=list)
    last_symptoms: list = field(default_factory=list)
    last_drug: str | None = None
    crisis_turns_left: int = 0

    @property
    def last_disease(self) -> str | None:
        return self.focus_diseases[0] if self.focus_diseases else None

    def prompt_block(self) -> str:
        if not self.turns and not self.focus_diseases:
            return "（这是对话的开始，还没有历史）"
        lines = []
        if self.focus_diseases:
            lines.append("当前关注的疾病：" + "、".join(self.focus_diseases))
        if self.last_symptoms:
            lines.append("用户提到的症状：" + "、".join(self.last_symptoms))
        if self.last_drug:
            lines.append(f"用户提到的药品：{self.last_drug}")
        if self.crisis_turns_left:
            lines.append("近期出现过危机表述，回复时要先确认人身安全。")
        for index, turn in enumerate(self.turns[-4:], 1):
            lines.append(f"{index}. 用户：{turn.query}")
            if turn.answer_brief:
                lines.append(f"   助手摘要：{turn.answer_brief}")
        return "\n".join(lines)

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
        self.turns.append(
            TurnRecord(
                query=query,
                answer_brief=brief,
                diseases=list(self.focus_diseases),
                intents=list(getattr(turn, "intents", []) or []),
            )
        )
        self.turns = self.turns[-8:]
