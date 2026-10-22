"""基于实体词典的多模式匹配。不依赖 BERT 权重，模型缺失时仍能识别词典中的实体。"""

import os

import ahocorasick

from agent.types import Perception


def _choose_type(types: set, disease_names: set) -> str:
    """同一个词既是疾病又是症状时，句中若还有别的疾病名，则把它当作症状。"""
    if "疾病" in types and "疾病症状" in types and len(disease_names) > 1:
        return "疾病症状"
    priority = ["疾病", "药品", "疾病症状", "治疗方法", "检查项目", "科目", "食物", "药品商"]
    for etype in priority:
        if etype in types:
            return etype
    return next(iter(types))


def perception_from_mentions(mentions: list, backend: str = "dictionary") -> Perception:
    by_type = {}
    for mention in mentions:
        bucket = by_type.setdefault(mention["type"], [])
        if mention["text"] not in bucket:
            bucket.append(mention["text"])
    diseases = by_type.get("疾病", [])
    symptoms = by_type.get("疾病症状", [])
    drugs = by_type.get("药品", [])
    return Perception(
        mentions=mentions,
        disease=diseases[0] if diseases else None,
        symptoms=list(symptoms),
        drug=drugs[0] if drugs else None,
        by_type=by_type,
        backend=backend,
    )


class DictionaryRecognizer:
    def __init__(self, entity_dir: str):
        self.entity_dir = entity_dir
        self.automata = {}
        if not os.path.isdir(entity_dir):
            return
        for filename in sorted(os.listdir(entity_dir)):
            if not filename.endswith(".txt"):
                continue
            etype = filename[:-4]
            automaton = ahocorasick.Automaton()
            path = os.path.join(entity_dir, filename)
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    word = line.strip().split(" ")[0]
                    if len(word) < 2:
                        continue
                    automaton.add_word(word, word)
            automaton.make_automaton()
            self.automata[etype] = automaton

    def find(self, text: str) -> list:
        if not text:
            return []
        grouped = {}
        for etype, automaton in self.automata.items():
            for end, word in automaton.iter(text):
                start = end - len(word) + 1
                grouped.setdefault((start, end, word), set()).add(etype)

        disease_names = {
            word for (_, _, word), types in grouped.items() if "疾病" in types
        }
        hits = []
        for (start, end, word), types in grouped.items():
            etype = _choose_type(types, disease_names)
            hits.append((start, end, etype, word))
        hits.sort(key=lambda item: len(item[3]), reverse=True)
        occupied = set()
        chosen = []
        for start, end, etype, word in hits:
            span = range(start, end + 1)
            if any(index in occupied for index in span):
                continue
            chosen.append({"start": start, "end": end, "type": etype, "text": word})
            occupied.update(span)
        chosen.sort(key=lambda item: item["start"])
        return chosen

    def recognize(self, text: str) -> Perception:
        return perception_from_mentions(self.find(text), backend="dictionary")


class HybridRecognizer:
    """词典负责多实体召回，BERT 补上词典没覆盖的说法。"""

    def __init__(self, dictionary: DictionaryRecognizer, bert_align):
        self.dictionary = dictionary
        self.bert_align = bert_align

    def recognize(self, text: str) -> Perception:
        perception = self.dictionary.recognize(text)
        try:
            extra = self.bert_align(text) or {}
        except Exception:
            extra = {}
        mentions = list(perception.mentions)
        by_type = {key: list(value) for key, value in perception.by_type.items()}
        for etype, name in extra.items():
            if not name:
                continue
            bucket = by_type.setdefault(etype, [])
            if name not in bucket:
                bucket.append(name)
                mentions.append({"start": -1, "end": -1, "type": etype, "text": name})
        merged = perception_from_mentions(mentions, backend="bert+dictionary")
        merged.by_type = by_type
        merged.disease = by_type.get("疾病", [None])[0]
        merged.symptoms = list(by_type.get("疾病症状", []))
        merged.drug = by_type.get("药品", [None])[0]
        return merged
