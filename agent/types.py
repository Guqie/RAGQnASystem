from dataclasses import dataclass, field


@dataclass
class Perception:
    """一轮问句的实体感知结果。"""

    mentions: list
    disease: str | None = None
    symptoms: list = field(default_factory=list)
    drug: str | None = None
    by_type: dict = field(default_factory=dict)
    backend: str = "dictionary"

    @staticmethod
    def empty() -> "Perception":
        return Perception(mentions=[])

    @property
    def diseases(self) -> list:
        return list(self.by_type.get("疾病", []))

    def focus_names(self) -> list:
        names = []
        for name in self.diseases:
            if name and name not in names:
                names.append(name)
        if self.disease and self.disease not in names:
            names.insert(0, self.disease)
        return names


@dataclass
class Profile:
    disease: str
    text: str
    attributes: dict = field(default_factory=dict)
    relations: dict = field(default_factory=dict)
