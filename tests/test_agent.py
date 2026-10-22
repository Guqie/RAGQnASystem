import tempfile
import unittest
from pathlib import Path

from agent.core import CounselingAgent
from agent.dictionary import DictionaryRecognizer
from agent.knowledge import Neo4jKnowledge
from agent.memory import SessionMemory
from agent.planner import extract_json_object, normalize_plan, rule_plan, select_intents
from agent.safety import is_crisis
from agent.sample_kg import demo_knowledge
from agent.types import Perception


class Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class ScriptedGraph:
    def __init__(self):
        self.calls = []

    def run(self, cypher, **params):
        self.calls.append((cypher, params))
        if "IN $symptoms" in cypher:
            return Result([{"name": "抑郁症", "hit": 2}, {"name": "焦虑症", "hit": 1}])
        if "药品商" in cypher:
            return Result([{"name": "演示药厂"}])
        if "疾病简介" in cypher:
            return Result([{"简介": "心境障碍", "病因": "多因素", "易感人群": "成年人"}])
        if "DISTINCT n.名称" in cypher:
            return Result([{"name": "舍曲林"}])
        if "RETURN d.名称 AS name" in cypher:
            return Result([{"name": params.get("name")}])
        return Result([])


class MapRecognizer:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def recognize(self, text):
        return self.mapping.get(text, Perception.empty())


class FakeLLM:
    def __init__(self, content):
        self.content = content

    def complete(self, messages, **kwargs):
        return self.content


class BoomKnowledge:
    available = True

    def resolve_disease(self, name):
        raise AssertionError("危机轮次不应解析疾病")

    def query(self, *args, **kwargs):
        raise AssertionError("危机轮次不应查询图谱")

    def diseases_by_symptoms(self, *args, **kwargs):
        raise AssertionError("危机轮次不应做症状推断")

    def drug_producers(self, *args, **kwargs):
        raise AssertionError("危机轮次不应查询药品")


class AgentTests(unittest.TestCase):
    def test_crisis_detects_distress_without_flagging_negation(self):
        self.assertTrue(is_crisis("我不想活了"))
        self.assertTrue(is_crisis("最近总想死"))
        self.assertFalse(is_crisis("我其实不想死，只是很累"))
        self.assertFalse(is_crisis("抑郁症怎么治疗"))

    def test_crisis_turn_skips_tools_and_keeps_hotline(self):
        agent = CounselingAgent(BoomKnowledge(), MapRecognizer(), llm=None)
        memory = SessionMemory()
        memory.focus_diseases = ["抑郁症"]
        turn = agent.step("我不想活了", memory)
        self.assertTrue(turn.crisis)
        self.assertIn("12356", turn.final_text)
        self.assertNotIn("舍曲林", turn.final_text)
        memory.observe("我不想活了", turn.final_text, turn)
        self.assertEqual(memory.last_disease, "抑郁症")
        self.assertEqual(memory.crisis_turns_left, 2)

    def test_followup_reuses_disease_and_queries_drugs(self):
        first = "我最近情绪低落，还失眠，这是抑郁症吗？"
        second = "那要吃药吗？"
        recognizer = MapRecognizer(
            {
                first: Perception(
                    mentions=[],
                    disease="抑郁症",
                    symptoms=["情绪低落", "失眠"],
                    by_type={"疾病": ["抑郁症"], "疾病症状": ["情绪低落", "失眠"]},
                )
            }
        )
        agent = CounselingAgent(demo_knowledge(), recognizer, llm=None)
        memory = SessionMemory()
        first_turn = agent.step(first, memory)
        self.assertEqual(first_turn.resolved_diseases, ["抑郁症"])
        self.assertIn("心境障碍", first_turn.knowledge)
        memory.observe(first, first_turn.final_text, first_turn)

        second_turn = agent.step(second, memory)
        self.assertEqual(second_turn.resolved_diseases, ["抑郁症"])
        self.assertIn("舍曲林", second_turn.knowledge)
        self.assertIn("沿用上一轮", second_turn.plan["reasoning"])
        self.assertTrue(second_turn.plan["use_memory"])

    def test_symptoms_without_disease_infer_then_lookup(self):
        query = "总是失眠，还情绪低落"
        recognizer = MapRecognizer(
            {
                query: Perception(
                    mentions=[],
                    symptoms=["失眠", "情绪低落"],
                    by_type={"疾病症状": ["失眠", "情绪低落"]},
                )
            }
        )
        turn = CounselingAgent(demo_knowledge(), recognizer, llm=None).step(query, SessionMemory())
        self.assertEqual(turn.resolved_diseases[0], "抑郁症")
        self.assertTrue(any("症状推断" in item["stage"] for item in turn.trace))
        self.assertIn("不是诊断", turn.knowledge)
        self.assertIn("兴趣下降", turn.knowledge)

    def test_greeting_does_not_dump_previous_disease_facts(self):
        memory = SessionMemory()
        memory.focus_diseases = ["抑郁症"]
        memory.turns.append(type("T", (), {"query": "之前", "answer_brief": "聊过"})())
        turn = CounselingAgent(demo_knowledge(), MapRecognizer(), llm=None).step("你好", memory)
        self.assertEqual(turn.plan["actions"][0]["tool"], "counsel_only")
        self.assertNotIn("舍曲林", turn.knowledge)

    def test_llm_plan_is_used_when_json_is_valid(self):
        raw = """思考如下：
```json
{"reasoning":"问生产商","use_memory":false,"intents":["查询药品的生产商","不存在的意图"],"actions":[{"tool":"query_drug","drug":"舍曲林"}]}
```
"""
        query = "舍曲林是哪家生产的"
        recognizer = MapRecognizer(
            {query: Perception(mentions=[], drug="舍曲林", by_type={"药品": ["舍曲林"]})}
        )
        turn = CounselingAgent(demo_knowledge(), recognizer, FakeLLM(raw)).step(query, SessionMemory())
        self.assertEqual(turn.plan["source"], "llm")
        self.assertIsNone(turn.final_text)
        self.assertIn("演示药厂", turn.knowledge)
        self.assertIn("演示药厂", turn.messages[1]["content"])
        self.assertNotIn("不存在的意图", turn.plan["intents"])

    def test_invalid_llm_plan_falls_back_to_rules(self):
        query = "我最近情绪低落，还失眠，这是抑郁症吗？"
        recognizer = MapRecognizer(
            {
                query: Perception(
                    mentions=[],
                    disease="抑郁症",
                    symptoms=["失眠"],
                    by_type={"疾病": ["抑郁症"], "疾病症状": ["失眠"]},
                )
            }
        )
        turn = CounselingAgent(demo_knowledge(), recognizer, FakeLLM("我不会输出 JSON")).step(
            query, SessionMemory()
        )
        self.assertEqual(turn.plan["source"], "rule")
        self.assertIn("规则回退", [item["status"] for item in turn.trace])
        self.assertEqual(turn.resolved_diseases, ["抑郁症"])

    def test_missing_graph_skips_execution_but_keeps_plan(self):
        query = "抑郁症怎么治疗"
        recognizer = MapRecognizer(
            {query: Perception(mentions=[], disease="抑郁症", by_type={"疾病": ["抑郁症"]})}
        )
        turn = CounselingAgent(Neo4jKnowledge(None), recognizer, llm=None).step(query, SessionMemory())
        self.assertIn("没有连上", turn.knowledge)
        self.assertTrue(any(item["status"] == "跳过" for item in turn.trace))
        self.assertIn("治疗", turn.plan["reasoning"])

    def test_cypher_matches_graph_property_names(self):
        graph = ScriptedGraph()
        knowledge = Neo4jKnowledge(graph)
        self.assertEqual(knowledge.resolve_disease("抑郁症"), "抑郁症")
        profile = knowledge.query(
            "抑郁症",
            ["查询疾病简介", "查询疾病病因", "查询疾病易感人群", "查询疾病常用药品"],
        )
        cypher = "\n".join(call[0] for call in graph.calls)
        self.assertIn("d.`疾病简介`", cypher)
        self.assertIn("d.`疾病病因`", cypher)
        self.assertIn("d.`疾病易感人群`", cypher)
        self.assertIn("疾病使用药品", cypher)
        self.assertIn("舍曲林", profile.text)
        rows = knowledge.diseases_by_symptoms(["失眠", "情绪低落"])
        self.assertEqual(rows[0]["name"], "抑郁症")
        self.assertEqual(graph.calls[-1][1]["symptoms"], ["失眠", "情绪低落"])

    def test_rule_planner_limits_intents_and_parses_messy_json(self):
        intents = select_intents("那要吃药吗？怎么治疗，要多久")
        self.assertLessEqual(len(intents), 3)
        self.assertIn("查询疾病常用药品", intents)
        self.assertIn("查询疾病的治疗方法", intents)
        parsed = extract_json_object('前文 {"reasoning":"ok","actions":[{"tool":"counsel_only"}]} 后文')
        plan = normalize_plan(parsed)
        self.assertEqual(plan["actions"][0]["tool"], "counsel_only")
        self.assertIsNone(normalize_plan({"actions": [{"tool": "delete_database"}]}))

    def test_dictionary_prefers_longer_entity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "疾病.txt").write_text("抑郁症\n失眠抑郁\n焦虑症\n", encoding="utf-8")
            (root / "疾病症状.txt").write_text("失眠\n情绪低落\n", encoding="utf-8")
            (root / "药品.txt").write_text("舍曲林\n", encoding="utf-8")
            recognizer = DictionaryRecognizer(str(root))
            overlapped = recognizer.recognize("我有失眠抑郁，还情绪低落")
            self.assertEqual(overlapped.disease, "失眠抑郁")
            self.assertEqual(overlapped.symptoms, ["情绪低落"])
            both = recognizer.recognize("我得了抑郁症，晚上失眠")
            self.assertEqual(both.disease, "抑郁症")
            self.assertEqual(both.symptoms, ["失眠"])
            (root / "疾病.txt").write_text("抑郁症\n失眠\n", encoding="utf-8")
            ambiguous = DictionaryRecognizer(str(root)).recognize("总是失眠，这是抑郁症吗")
            self.assertEqual(ambiguous.diseases, ["抑郁症"])
            self.assertIn("失眠", ambiguous.symptoms)

    def test_unresolved_disease_is_traced_but_not_shown_as_fact(self):
        query = "失眠怎么办"
        recognizer = MapRecognizer(
            {query: Perception(mentions=[], disease="失眠", by_type={"疾病": ["失眠"]})}
        )
        turn = CounselingAgent(demo_knowledge(), recognizer, llm=None).step(query, SessionMemory())
        self.assertTrue(any(item["status"] == "未命中" for item in turn.trace))
        self.assertNotIn("没有在图谱中定位", turn.knowledge)
        self.assertNotIn("没有在图谱中定位", turn.final_text)

    def test_memory_prompt_mentions_recent_focus(self):
        memory = SessionMemory()
        turn = type(
            "Turn",
            (),
            {
                "crisis": False,
                "resolved_diseases": ["焦虑症"],
                "symptoms": ["心慌"],
                "drug": None,
                "intents": ["查询疾病简介"],
            },
        )()
        memory.observe("我是不是焦虑症", "先别急着给自己下结论。", turn)
        text = memory.prompt_block()
        self.assertIn("焦虑症", text)
        self.assertIn("心慌", text)
        self.assertIn("我是不是焦虑症", text)


if __name__ == "__main__":
    unittest.main()
