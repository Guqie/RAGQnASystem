"""疾病文档库。图谱负责结构化关系，这里保留简介、病因等原文，供另一路检索。"""

import json
import math
import os
import pickle
import re
from pathlib import Path


def tokenize(text: str) -> list:
    compact = re.sub(r"\s+", "", str(text).lower())
    if len(compact) < 2:
        return [compact] if compact else []
    return [compact[i : i + 2] for i in range(len(compact) - 1)]


class DocumentStore:
    def __init__(self):
        self.passages = []
        self._pending = []
        self._postings = {}
        self.avgdl = 1.0
        self.ready = False

    def add(self, disease: str, section: str, text: str) -> None:
        body = " ".join(str(text).split())
        if not disease or not body:
            return
        tokens = tokenize(f"{disease}{section}{body}")
        if not tokens:
            return
        frequencies = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        self._pending.append((disease, section, body[:360], len(tokens), frequencies))
        self.ready = False

    def finalize(self) -> None:
        self.passages = []
        self._postings = {}
        for disease, section, body, length, frequencies in self._pending:
            doc_id = len(self.passages)
            self.passages.append({"disease": disease, "section": section, "text": body, "len": length})
            for token, freq in frequencies.items():
                self._postings.setdefault(token, []).append((doc_id, freq))
        total = sum(item["len"] for item in self.passages)
        self.avgdl = (total / len(self.passages)) if self.passages else 1.0
        self.ready = True

    def search(self, question: str, k: int = 3) -> list:
        if not self.ready:
            self.finalize()
        if not self.passages or not question:
            return []
        scores = {}
        total_docs = len(self.passages)
        for token in set(tokenize(question)):
            postings = self._postings.get(token)
            if not postings:
                continue
            idf = math.log(1 + (total_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc_id, freq in postings:
                length = self.passages[doc_id]["len"] or 1
                denom = freq + 1.2 * (1 - 0.75 + 0.75 * length / self.avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * freq * 2.2 / denom
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        found = []
        for doc_id, score in ranked:
            passage = dict(self.passages[doc_id])
            passage["score"] = round(score, 4)
            found.append(passage)
        return found

    @staticmethod
    def from_records(records: dict) -> "DocumentStore":
        store = DocumentStore()
        for disease, record in records.items():
            for section, text in (record.get("attributes") or {}).items():
                store.add(disease, section, text)
            for section, items in (record.get("relations") or {}).items():
                store.add(disease, section, "、".join(str(item) for item in items))
        store.finalize()
        return store


def load_medical_jsonl(path: str, snippet: int = 160) -> DocumentStore:
    store = DocumentStore()
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip().rstrip(",")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = obj.get("name") or ""
            if not name:
                continue
            desc = (obj.get("desc") or "")[:snippet]
            cause = (obj.get("cause") or "")[:snippet]
            if desc:
                store.add(name, "简介", desc)
            if cause:
                store.add(name, "病因", cause)
    store.finalize()
    return store


def cached_medical_store(source: str, cache_path: str) -> DocumentStore:
    source_path = Path(source)
    cache = Path(cache_path)
    stamp = os.path.getmtime(source_path)
    if cache.exists():
        try:
            with cache.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("mtime") == stamp and isinstance(payload.get("store"), DocumentStore):
                return payload["store"]
        except (OSError, pickle.UnpicklingError, EOFError):
            pass
    store = load_medical_jsonl(str(source_path))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as handle:
        pickle.dump({"mtime": stamp, "store": store}, handle)
    return store


class LazyDocuments:
    """第一次检索时才构建文档库，避免演示和测试去读完整医疗语料。"""

    def __init__(self, loader):
        self._loader = loader
        self._store = None

    def search(self, question: str, k: int = 3) -> list:
        if self._store is None:
            self._store = self._loader()
        return self._store.search(question, k=k)


def format_passages(passages: list) -> str:
    if not passages:
        return ""
    lines = ["【文档】"]
    for item in passages:
        lines.append(f"[{item['disease']} · {item['section']}] {item['text']}")
    return "\n".join(lines)
