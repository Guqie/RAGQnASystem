"""离线走一遍 Agent 循环，不需要 Neo4j，也不需要 API Key。

    python -m agent
"""

from agent.bert_ner import recognizer_backend
from agent.config import settings
from agent.core import CounselingAgent
from agent.dictionary import DictionaryRecognizer
from agent.memory import SessionMemory
from agent.sample_kg import demo_knowledge
from agent.types import Perception


class _KeywordRecognizer:
    """词典文件不可用时的最小识别器，只覆盖演示问句。"""

    lexicon = {
        "疾病": ["抑郁症", "焦虑症"],
        "疾病症状": ["情绪低落", "失眠", "心慌"],
        "药品": ["舍曲林"],
    }

    def recognize(self, text: str) -> Perception:
        mentions = []
        cursor = 0
        for etype, words in self.lexicon.items():
            for word in words:
                start = text.find(word)
                if start >= 0:
                    mentions.append({"start": start, "end": start + len(word) - 1, "type": etype, "text": word})
                    cursor += 1
        mentions.sort(key=lambda item: item["start"])
        from agent.dictionary import perception_from_mentions

        return perception_from_mentions(mentions, backend="demo-keywords")


def build_demo_recognizer():
    if settings.entity_dir.is_dir() and any(settings.entity_dir.glob("*.txt")):
        return DictionaryRecognizer(str(settings.entity_dir))
    return _KeywordRecognizer()


def run_dialogue(agent: CounselingAgent, queries: list) -> SessionMemory:
    memory = SessionMemory()
    for query in queries:
        turn = agent.step(query, memory)
        print("=" * 60)
        print(f"用户：{query}")
        for item in turn.trace:
            print(f"\n[{item['stage']} · {item['status']}]")
            print(item["detail"])
        print("\n[回复]")
        print(turn.final_text or "（已生成提示词，等待大模型）")
        memory.observe(query, turn.final_text or "", turn)
    return memory


def main():
    recognizer = build_demo_recognizer()
    agent = CounselingAgent(demo_knowledge(), recognizer, llm=None)
    print("心理咨询 Agent 离线演示")
    print(f"实体识别：{recognizer_backend(recognizer)}")
    print("图谱：内置演示数据，不是完整的 Neo4j 医疗图谱。")
    queries = [
        "我最近情绪低落，总是失眠，这是抑郁症吗？",
        "那应该怎么治疗，要吃药吗？",
        "我不想活了",
    ]
    run_dialogue(agent, queries)


if __name__ == "__main__":
    main()
