import tempfile
import unittest
from pathlib import Path

from agent.core import CounselingAgent
from agent.dictionary import DictionaryRecognizer
from agent.knowledge import Neo4jKnowledge
from agent.llm import ModelMessage, ToolCall
from agent.memory import SessionMemory, load_session, save_session
from agent.nl2query import compile_statements, retrieval_text, run_natural_query
from agent.rag import DocumentStore
from agent.safety import is_crisis
from agent.sample_kg import demo_documents, demo_knowledge
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


class ToolLLM:
    def __init__(self, calls=None, chunks=None, explode=False):
        self.calls = calls or []
        self.chunks = chunks or ["资料里的生产商是", "演示药厂。"]
        self.explode = explode

    def complete_with_tools(self, messages, tools, **kwargs):
        if self.explode:
            raise RuntimeError("工具接口不可用")
        return ModelMessage(tool_calls=self.calls)

    def stream(self, messages, **kwargs):
        for chunk in self.chunks:
            yield chunk


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
        agent = CounselingAgent(demo_knowledge(), recognizer, llm=None, documents=demo_documents())
        memory = SessionMemory()
        first_turn = agent.step(first, memory)
        self.assertEqual(first_turn.resolved_diseases, ["抑郁症"])
        self.assertIn("1 抑郁症 疾病简介", first_turn.plan["statements"])
        self.assertIn("心境障碍", first_turn.knowledge)
        self.assertIn("【文档】", first_turn.knowledge)
        memory.observe(first, first_turn.final_text, first_turn)

        second_turn = agent.step(second, memory)
        self.assertEqual(second_turn.resolved_diseases, ["抑郁症"])
        self.assertIn("2 抑郁症 疾病使用药品 药品", second_turn.plan["statements"])
        self.assertIn("舍曲林", second_turn.knowledge)
        self.assertIn("舍曲林", second_turn.knowledge.split("【文档】", 1)[-1])
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
        turn = CounselingAgent(demo_knowledge(), recognizer, llm=None, documents=demo_documents()).step(
            query, SessionMemory()
        )
        self.assertEqual(turn.resolved_diseases[0], "抑郁症")
        self.assertTrue(any("症状推断" in item["stage"] for item in turn.trace))
        self.assertIn("不是诊断", turn.knowledge)
        self.assertIn("兴趣下降", turn.knowledge)
        self.assertIn("1 抑郁症 疾病简介", turn.plan["statements"])
        document_trace = next(item for item in turn.trace if item["stage"] == "工具 · 文档检索")
        rewritten = document_trace["detail"].split("检索问句：", 1)[1]
        self.assertIn("抑郁症", rewritten)
        self.assertIn("抑郁症", turn.knowledge.split("【文档】", 1)[1])

    def test_greeting_does_not_dump_previous_disease_facts(self):
        memory = SessionMemory()
        memory.focus_diseases = ["抑郁症"]
        memory.turns.append(type("T", (), {"query": "之前", "answer_brief": "聊过"})())
        turn = CounselingAgent(demo_knowledge(), MapRecognizer(), llm=None).step("你好", memory)
        self.assertEqual(turn.plan["tools"], [])
        self.assertNotIn("舍曲林", turn.knowledge)
        self.assertNotIn("舍曲林", turn.final_text)

    def test_model_tool_call_queries_graph_then_streams(self):
        query = "舍曲林是哪家生产的"
        recognizer = MapRecognizer(
            {query: Perception(mentions=[], drug="舍曲林", by_type={"药品": ["舍曲林"]})}
        )
        llm = ToolLLM(
            calls=[ToolCall("call-1", "query_knowledge_graph", '{"question":"舍曲林是哪家生产的","drug":"舍曲林"}')]
        )
        turn = CounselingAgent(demo_knowledge(), recognizer, llm).step(query, SessionMemory())
        self.assertEqual(turn.plan["source"], "llm")
        self.assertIn("query_knowledge_graph", turn.plan["tools"])
        self.assertIn("2 舍曲林 生产 药品商", turn.plan["statements"])
        self.assertIn("演示药厂", turn.knowledge)
        self.assertIn("演示药厂", turn.final_text)
        self.assertIn("演示药厂", turn.messages[1]["content"])

    def test_tool_call_failure_falls_back_to_local_query(self):
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
        turn = CounselingAgent(
            demo_knowledge(),
            recognizer,
            ToolLLM(explode=True, chunks=["已按本地查询整理回复。"]),
        ).step(query, SessionMemory())
        self.assertIn("回退", [item["status"] for item in turn.trace])
        self.assertEqual(turn.resolved_diseases, ["抑郁症"])
        self.assertIn("心境障碍", turn.knowledge)

    def test_missing_graph_skips_execution_but_keeps_plan(self):
        query = "抑郁症怎么治疗"
        recognizer = MapRecognizer(
            {query: Perception(mentions=[], disease="抑郁症", by_type={"疾病": ["抑郁症"]})}
        )
        turn = CounselingAgent(Neo4jKnowledge(None), recognizer, llm=None).step(query, SessionMemory())
        self.assertIn("没有连上", turn.knowledge)
        self.assertIn("2 抑郁症 治疗的方法 治疗方法", turn.plan["statements"])
        self.assertTrue(any(item["status"] == "跳过" for item in turn.trace))
        self.assertTrue(any("查询语句" in item["detail"] for item in turn.trace))

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
        graph.calls.clear()
        result = run_natural_query("抑郁症的病因是什么", ["抑郁症"], [], None, knowledge)
        self.assertIn("1 抑郁症 疾病病因", result.statements_text)
        cypher = "\n".join(call[0] for call in graph.calls)
        self.assertIn("d.`疾病病因`", cypher)

    def test_natural_query_caps_statements(self):
        statements = compile_statements("那要吃药吗？怎么治疗，要多久", ["抑郁症", "焦虑症"], [], None)
        fields = {item.field for item in statements}
        self.assertIn("疾病使用药品", fields)
        self.assertIn("治疗的方法", fields)
        self.assertLessEqual(len(statements), 4)
        rendered = [item.format() for item in statements]
        self.assertIn("2 抑郁症 疾病使用药品 药品", rendered)
        self.assertIn("1 抑郁症 治疗周期", rendered)
        shared = retrieval_text("那要吃药吗", statements)
        self.assertIn("抑郁症", shared)
        self.assertIn("疾病使用药品", shared)

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

    def test_document_search_prefers_matching_disease(self):
        store = DocumentStore()
        store.add("抑郁症", "简介", "以显著而持久的情绪低落为主要表现的心境障碍")
        store.add("百日咳", "简介", "百日咳是一种阵发性痉挛性咳嗽")
        store.finalize()
        found = store.search("抑郁症的心境障碍简介", k=2)
        self.assertEqual(found[0]["disease"], "抑郁症")

    def test_long_context_recall_and_disk(self):
        memory = SessionMemory()
        blank = type(
            "Turn",
            (),
            {"crisis": False, "resolved_diseases": ["抑郁症"], "symptoms": [], "drug": None, "intents": []},
        )()
        for index in range(8):
            memory.observe(f"第{index}个问题", "当时在聊睡眠", blank)
        self.assertTrue(any("第0个问题" in line for line in memory.summary_lines))
        recent = "\n".join(item.query for item in memory.turns)
        self.assertNotIn("第0个问题", recent)
        self.assertIn("第7个问题", recent)
        self.assertIn("第0个问题", memory.recall("第0个问题"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "session.json"
            save_session(path, [{"messages": [{"role": "user", "content": "第7个问题"}], "memory": memory}], 0)
            loaded, active = load_session(path)
        self.assertEqual(active, 0)
        self.assertEqual(loaded[0]["messages"][0]["content"], "第7个问题")
        self.assertEqual(loaded[0]["memory"].focus_diseases, ["抑郁症"])
        self.assertIn("第0个问题", loaded[0]["memory"].recall("第0个问题"))

    def test_stream_emits_multiple_tokens(self):
        agent = CounselingAgent(demo_knowledge(), MapRecognizer(), llm=None)
        events = list(agent.stream("你好", SessionMemory()))
        tokens = [event for event in events if event["type"] == "token"]
        self.assertGreater(len(tokens), 1)
        self.assertTrue(any(event["type"] == "done" for event in events))
        self.assertIn("我在听", "".join(event["text"] for event in tokens))


if __name__ == "__main__":
    unittest.main()
