import os
import streamlit as st
import ner_model as zwk # 假设这个模块是存在的
import pickle
import py2neo
import re
import torch # torch在加载模型时直接使用
from transformers import BertTokenizer
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
import langchain
from langchain.prompts import PromptTemplate
import json
import traceback

# 配置LangChain日志
langchain.verbose = True

# --- 安全警告：硬编码API密钥 ---
DEEPSEEK_API_KEY_FROM_USER = "sk-675fc771e6544564843fd32f52b34a03" 
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com/v1"
# --- 安全警告结束 ---

# 确保模型路径正确且可访问
LOCAL_INTENTION_MODEL_PATH = "/mnt/workspace/.cache/modelscope/models/Qwen/Qwen2.5-3B-Instruct" 

DEEPSEEK_API_MODELS = {
    "DeepSeek-Chat": "deepseek-chat",
    "DeepSeek-Coder": "deepseek-reasoner",
}

# Neo4j 连接信息
NEO4J_URI = "bolt://0.tcp.jp.ngrok.io:10038" 
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password" 

# 预定义的查询意图列表，与知识图谱的Schema设计紧密关联
PREDEFINED_INTENTS = [
    "查询疾病简介",          # 对应疾病的简介属性
    "查询疾病病因",          # 对应疾病的病因属性
    "查询疾病预防措施",      # 对应疾病的预防措施属性
    "查询疾病治疗周期",      # 对应疾病的治疗周期属性
    "查询治愈概率",          # 对应疾病的治愈概率属性
    "查询疾病易感人群",      # 对应疾病的易感人群属性
    "查询疾病常用药品",      # 对应疾病与药品的关系
    "查询疾病推荐药品",      # 对应疾病与药品的关系
    "查询疾病宜吃食物",      # 对应疾病与食物的关系 (包含推荐食谱)
    "查询疾病忌吃食物",      # 对应疾病与食物的关系
    "查询疾病所需检查",      # 对应疾病与检查项目的关系
    "查询疾病所属科室",      # 对应疾病与科室的关系
    "查询疾病的症状",        # 对应疾病与症状的关系
    "查询疾病的治疗方法",    # 对应疾病与治疗方法的关系
    "查询疾病的并发疾病",    # 对应疾病与疾病的并发关系
    "查询药品的生产商"       # 对应药品与药品商的关系
]

# 用于Streamlit流式输出的自定义回调处理器
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text=""):
        self.container = container
        self.text = initial_text
        self.has_started_generating = False

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        if not self.has_started_generating:
            self.text = ""
            self.has_started_generating = True

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌") 

    def on_llm_end(self, response, **kwargs) -> None:
        self.container.markdown(self.text) 
        self.has_started_generating = False

# 问答核心链类，整合了NER、意图识别、KG查询和答案生成
class MedicalQAChain:
    def __init__(self, client, bert_components, device, intention_llm):
        self.client = client # Neo4j 数据库客户端
        
        # BERT NER 模型组件
        if bert_components:
            self.bert_tokenizer = bert_components['tokenizer']
            self.bert_model = bert_components['model']
            self.idx2tag = bert_components['idx2tag']
            self.rule = bert_components['rule']
            self.tfidf_r = bert_components['tfidf_r']
            self.device = device
        else: 
            self.bert_tokenizer = None
            self.bert_model = None
            st.warning("BERT NER 组件未能完全加载，实体识别功能将受限。")

        self.intention_llm = intention_llm # Qwen 意图识别模型
        self.intention_prompt_template = self._create_intention_prompt() # 意图识别Prompt
        self.answer_prompt_template = self._create_answer_prompt() # 答案生成Prompt
        
        # 意图到Cypher查询的映射字典，用于模块化知识图谱检索
        self.intent_to_cypher_map = self._create_intent_to_cypher_map()

    # 创建意图识别的Prompt模板
    def _create_intention_prompt(self):
        intent_categories_str = "\n".join([f"- \"{intent}\"" for intent in PREDEFINED_INTENTS])
        template = f"""你是一名专业的心理咨询问答助手，擅长理解用户问题并识别其核心意图。
        你的任务是分析用户提问，并从中识别出**一个或多个**最相关的查询意图。
        你必须从以下预定义的查询类别中选择。

        **输出格式要求：**
        你的回答必须是一个**纯粹的JSON列表**，例如 `["意图1", "意图2"]`。
        **严格遵守此格式。除JSON列表外，不要输出任何额外的文字、解释或多余的JSON结构。**
        输出的意图数量至多不超过3个。

        **预定义的查询类别：**
        {intent_categories_str}

        以下是一些在心理咨询场景中，识别意图的示例：
        **示例1：**
        输入："我最近情绪很低落，总是提不起精神，这是抑郁症吗？"
        输出：["查询疾病简介", "查询疾病的症状"]

        **示例2：**
        输入："我朋友得了焦虑症，他应该怎么治疗才好？"
        输出：["查询疾病的治疗方法", "查询疾病常用药品"]

        **示例3：**
        输入："如果我得了双相情感障碍，日常生活中要注意些什么？"
        输出：["查询疾病预防措施", "查询疾病宜吃食物", "查询疾病忌吃食物"]

        **示例4：**
        输入："我感觉压力很大，总是失眠，这是什么原因？"
        输出：["查询疾病简介", "查询疾病病因", "查询疾病的症状"]

        **示例5：**
        输入："孩子有点多动症，需要做什么检查才能确诊？"
        输出：["查询疾病所需检查", "查询疾病简介"]

        **示例6：**
        输入："抑郁症一般要多久才能好？治疗效果怎么样？"
        输出：["查询疾病治疗周期", "查询治愈概率"]

        **示例7：**
        输入："得了强迫症会引发其他精神问题吗？"
        输出：["查询疾病的并发疾病", "查询疾病简介"]
        **注意**：
        - 输出必须在预定义类别中
        - 输出的类别数量至多不超过3个
        - 每个类别后添加注释解释原因

        输入：{{query}}
        输出：
        """
        return PromptTemplate.from_template(template)

    # 创建答案生成的Prompt模板
    def _create_answer_prompt(self):
        template = """<指令>你是一名专业、富有同情心和耐心的心理咨询师。你的任务是倾听用户的困扰，并根据他们的提问提供支持、理解和引导。在你的回复中，请始终秉持以下结构化流程：
    1.  **共情与确认**：首先，对用户的情绪和处境表达理解与关心，确认他们正在经历的感受。
    2.  **梳理与澄清**：帮助用户理清问题的核心，可能涉及对问题的初步界定或澄清。
    3.  **结合背景信息（若有）**：如果提供了相关的背景知识或医疗信息（来自知识图谱），请巧妙地将其融入到你的咨询话语中，作为理解用户情况或提供建议的参考。避免直接罗列知识，而是以咨询师的口吻进行解释或引导。
    4.  **提供支持与应对思路**：基于你对用户困扰的理解，以及（若有）的背景知识，给出一些初步的应对策略、思考方向或自我探索的问题，帮助用户找到解决问题的可能性。这些建议应是温和、可操作且符合心理咨询原则的。
    5.  **鼓励与未来展望**：表达对用户的支持和鼓励，并邀请他们继续沟通，以便更深入地探讨或调整方案。
    <注意>
    **关于知识库信息：**
    - 如果以下`<知识库信息>`标签中的内容是“知识库中没有找到与您问题直接相关的具体医学信息。不过，别担心，我会尽力从心理支持的角度给您一些建议。”，这意味着我们的医疗知识图谱暂时无法提供直接的医学参考。在这种情况下，请你**专注于作为心理咨询师的角色**，根据用户的原始问题，主要按照上述**1、2、4、5步骤**提供纯心理咨询的支持和引导。你可以简单提及知识库信息有限，然后转向心理支持。
    - 否则，请你将`<知识库信息>`中的内容作为**重要的参考资料**，以你心理咨询师的专业视角进行解读和整合，自然地融入到你的回复中。
    用户提问：{query}
    <知识库信息>
    {prompt}
    </知识库信息>
    请你严格遵循上述指令和结构，以自然、流畅的中文口语表达进行回复。避免使用生硬的专业术语列表。始终保持同理心和专业性的平衡。
    </注意></指令>"""
        return PromptTemplate.from_template(template)
    
    # 意图到Cypher查询的映射字典，定义了每种意图对应的查询模板和所需实体类型
    def _create_intent_to_cypher_map(self):
        return {
            "查询疾病简介": ("MATCH (d:疾病 {名称: $entity_name}) RETURN d.desc AS value", "disease"),
            "查询疾病病因": ("MATCH (d:疾病 {名称: $entity_name}) RETURN d.病因 AS value", "disease"),
            "查询疾病预防措施": ("MATCH (d:疾病 {名称: $entity_name}) RETURN d.预防措施 AS value", "disease"),
            "查询疾病治疗周期": ("MATCH (d:疾病 {名称: $entity_name}) RETURN d.治疗周期 AS value", "disease"),
            "查询治愈概率": ("MATCH (d:疾病 {名称: $entity_name}) RETURN d.治愈概率 AS value", "disease"),
            "查询疾病易感人群": ("MATCH (d:疾病 {名称: $entity_name}) RETURN d.易感人群 AS value", "disease"),
            "查询疾病常用药品": ("MATCH (d:疾病 {名称: $entity_name})-[:疾病使用药品]->(drug:药品) RETURN drug.名称 AS 名称", "disease"),
            "查询疾病推荐药品": ("MATCH (d:疾病 {名称: $entity_name})-[:疾病推荐药品]->(drug:药品) RETURN drug.名称 AS 名称", "disease"),
            "查询疾病忌吃食物": ("MATCH (d:疾病 {名称: $entity_name})-[:疾病忌吃食物]->(f:食物) RETURN f.名称 AS 名称", "disease"),
            "查询疾病所需检查": ("MATCH (d:疾病 {名称: $entity_name})-[:疾病所需检查]->(c:检查项目) RETURN c.名称 AS 名称", "disease"),
            "查询疾病所属科室": ("MATCH (d:疾病 {名称: $entity_name})-[:疾病所属科目]->(dept:科目) RETURN dept.名称 AS 名称", "disease"),
            "查询疾病的症状": ("MATCH (d:疾病 {名称: $entity_name})-[:疾病的症状]->(s:疾病症状) RETURN s.名称 AS 名称", "disease"),
            "查询疾病的治疗方法": ("MATCH (d:疾病 {名称: $entity_name})-[:治疗的方法]->(cure:治疗方法) RETURN cure.名称 AS 名称", "disease"),
            "查询疾病的并发疾病": ("MATCH (d:疾病 {名称: $entity_name})-[:疾病并发疾病]->(ad:疾病) RETURN ad.名称 AS 名称", "disease"),
            "查询药品的生产商": ("MATCH (p:药品商)-[:生产]->(dr:药品 {名称: $entity_name}) RETURN p.名称 AS 名称", "drug"),
            "查询疾病宜吃食物": ("custom_food_query", "disease") # 特殊意图处理，因为它可能需要合并多种关系
        }

    # 执行BERT实体识别
    def _get_ner_results(self, query):
        # 检查NER组件是否已加载
        if not all(hasattr(self, attr) and getattr(self, attr) is not None for attr in ['bert_model', 'bert_tokenizer', 'rule', 'tfidf_r', 'idx2tag']):
            st.warning("BERT NER 组件未能完全加载，实体识别功能将受限。")
            return {}
        try:
            return zwk.get_ner_result(
                self.bert_model, self.bert_tokenizer, query,
                self.rule, self.tfidf_r, self.device, self.idx2tag
            )
        except Exception as e:
            st.error(f"NER过程中出错: {e}")
            traceback.print_exc()
            return {}

    # 解析意图识别大模型的原始输出
    def _parse_intention_output(self, intention_output_str: str) -> list:
        print(f"DEBUG_PARSE: Received raw intention output: '''{intention_output_str}'''")
        
        search_space = intention_output_str

        # 1. 尝试从 ###开始输出### 和 ###结束输出### 之间提取
        content_between_markers_match = re.search(r"###开始输出###(.*?)###结束输出###", intention_output_str, re.DOTALL)
        if content_between_markers_match:
            search_space = content_between_markers_match.group(1).strip()
            print(f"DEBUG_PARSE: Searching within markers: '''{search_space}'''")
        else:
            # 2. 如果没有标记，尝试在 "输出：" 或 "你的输出（JSON列表）：" 之后查找
            output_marker_match = re.search(r"(?:你的输出（JSON列表）：|输出：)(.*)", intention_output_str, re.DOTALL)
            if output_marker_match:
                search_space = output_marker_match.group(1).strip()
                print(f"DEBUG_PARSE: Searching after output marker: '''{search_space}'''")
            else:
                # 3. 否则，在整个原始字符串中查找（作为兜底）
                print(f"DEBUG_PARSE: No specific markers found, searching full original string.")
                # search_space 此时仍是 intention_output_str

        # 查找所有可能的JSON列表
        json_list_candidates = re.findall(r"(\[(?:[^\[\]\"]|\"(?:\\\"|[^\"])*\")+?\])", search_space, re.DOTALL)
        
        if not json_list_candidates:
            print(f"DEBUG_PARSE: No JSON list candidates found in the search space.")
            # 尝试直接解析整个原始字符串作为兜底，可能本身就是个列表
            try:
                # 确保清理，避免影响直接解析
                cleaned_for_direct_parse = intention_output_str.replace("```json", "").replace("```", "").strip()
                parsed_list_direct = json.loads(cleaned_for_direct_parse)
                if isinstance(parsed_list_direct, list) and all(isinstance(item, str) for item in parsed_list_direct):
                    valid_intentions_direct = [item.strip().replace('\u200b', '') for item in parsed_list_direct if item.strip().replace('\u200b', '') in PREDEFINED_INTENTS]
                    if valid_intentions_direct:
                        print(f"DEBUG_PARSE: Found valid list by direct full string parse: {valid_intentions_direct[:3]}")
                        return valid_intentions_direct[:3]
            except json.JSONDecodeError:
                pass # 尝试失败，继续

        # 遍历所有找到的候选，寻找最合理的一个。从后往前遍历，倾向于最新生成的。
        for json_candidate_str in reversed(json_list_candidates):
            print(f"DEBUG_PARSE: Trying candidate (from reversed list): '''{json_candidate_str}'''")
            
            # 过滤掉包含Prompt关键词的候选，这些很可能是模型重复Prompt结构
            # 这是一个关键的过滤步骤，确保我们只解析模型实际“生成”的意图，而不是Prompt的重复部分
            if any(keyword in json_candidate_str for keyword in [
                "用户问题：", "你的输出：", "请直接输出JSON列表：", "严格的输出格式要求：", 
                "预定义的查询类别：", "示例（请严格模仿输出格式）：", "###开始输出###", "###结束输出###",
                "输出：", # 这个也要过滤，因为模型可能把"输出："也复制了
            ]):
                print(f"DEBUG_PARSE: Skipping candidate due to prompt keywords: '''{json_candidate_str}'''")
                continue
            
            try:
                parsed_list = json.loads(json_candidate_str)
                print(f"DEBUG_PARSE: JSON parsed to: {parsed_list}, type: {type(parsed_list)}")
                
                if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                    valid_intentions = []
                    for item in parsed_list:
                        # 清理意图字符串，移除可能的首尾空白和零宽空格
                        cleaned_item = item.strip().replace('\u200b', '') 
                        if cleaned_item in PREDEFINED_INTENTS:
                            valid_intentions.append(cleaned_item)
                        else:
                            print(f"DEBUG_PARSE: Item '{item}' (cleaned: '{cleaned_item}') not found in PREDEFINED_INTENTS. This candidate might be problematic.")
                            valid_intentions = [] # 如果有任何一个item不合法，则整个候选列表都可疑
                            break # 跳出内层循环，检查下一个json_candidate_str
                    
                    if valid_intentions:
                        print(f"DEBUG_PARSE: Successfully parsed and validated: {valid_intentions[:3]}")
                        return valid_intentions[:3] # 找到第一个有效且符合预期的就返回
                else:
                    print(f"DEBUG_PARSE: Parsed list is NOT valid (not list of strings).")
            except json.JSONDecodeError as e:
                print(f"DEBUG_PARSE: JSON parsing failed for '''{json_candidate_str}''': {e}")
        
        # 如果解析失败或未找到有效JSON，返回默认意图
        final_default_intentions = ["查询疾病专家建议"]
        st.warning(f"未能从以下输出中有效解析意图: '{intention_output_str}'. 使用默认意图。")
        print(f"DEBUG_PARSE_FINAL: Returning DEFAULT intentions: {final_default_intentions}")
        return final_default_intentions

    # 根据意图和实体生成知识图谱查询Prompt
    def _generate_knowledge_prompt(self, entities, parsed_intentions: list): # 接收解析后的意图列表
        prompt_parts = []
        yitu_matched_in_kg = []
        disease_entity_name = entities.get('疾病') 
        drug_entity_name = entities.get('药品')   
        symptom_entity_name = entities.get('疾病症状') 

        print(f"DEBUG_GEN_PROMPT: Received entities: {entities}")
        print(f"DEBUG_GEN_PROMPT: Received parsed_intentions: {parsed_intentions}, type: {type(parsed_intentions)}")

        # --- 症状反向查找疾病的逻辑 ---
        # 只有当意图明确需要疾病实体，但实体识别只提供了症状时才进行推断
        # 辅助函数：检查意图是否需要疾病实体
        def does_intent_need_disease(intent):
            return intent in [
                "查询疾病简介", "查询疾病病因", "查询疾病预防措施", "查询疾病治疗周期",
                "查询治愈概率", "查询疾病易感人群", "查询疾病常用药品", "查询疾病推荐药品",
                "查询疾病宜吃食物", "查询疾病忌吃食物", "查询疾病所需检查", "查询疾病所属科室",
                "查询疾病的症状", "查询疾病的治疗方法", "查询疾病的并发疾病"
            ]
        
        # 检查解析到的意图列表中是否有任何一个需要疾病实体的意图
        needs_disease_entity_from_intents = any(does_intent_need_disease(intent) for intent in parsed_intentions)

        if not disease_entity_name and symptom_entity_name and needs_disease_entity_from_intents:
            print(f"DEBUG_GEN_PROMPT: 识别到症状 '{symptom_entity_name}'，且意图需要疾病实体，尝试反向查找疾病。")
            try:
                related_disease_query = "MATCH (s:疾病症状 {名称: $symptom_name})<-[:疾病的症状]-(d:疾病) RETURN DISTINCT d.名称 AS disease_name LIMIT 1"
                res_diseases = self.client.run(related_disease_query, symptom_name=symptom_entity_name).data()
                
                if res_diseases:
                    inferred_disease_name = res_diseases[0]['disease_name']
                    disease_entity_name = inferred_disease_name # 更新 disease_entity_name，以便后续查询使用
                    prompt_parts.append(f"<提示>根据您描述的症状“{symptom_entity_name}”，我们推断可能与“{inferred_disease_name}”相关。</提示>")
                    print(f"DEBUG_GEN_PROMPT: 从症状 '{symptom_entity_name}' 推断出疾病: '{inferred_disease_name}'")
                else:
                    print(f"DEBUG_GEN_PROMPT: 未能从症状 '{symptom_entity_name}' 推断出相关疾病。")
            except Exception as e:
                print(f"ERROR: 症状反向查找疾病失败: {e}")
                traceback.print_exc()
        # --- 症状反向查找疾病的逻辑结束 ---

        # 辅助函数：格式化Cypher查询结果为列表字符串
        list_formatter = lambda r_list, key='名称': "、".join([item[key] for item in r_list if item and item.get(key)]) if r_list else None
        # 辅助函数：格式化Cypher查询结果为单个值
        single_value_formatter = lambda r, key='value': r[0][key] if r and r[0] and r[0].get(key) is not None else None

        # 遍历解析后的意图列表，根据意图调用对应的Cypher查询
        for intent in parsed_intentions:
            cypher_info = self.intent_to_cypher_map.get(intent)
            if not cypher_info:
                print(f"DEBUG_GEN_PROMPT: 未找到意图 '{intent}' 对应的Cypher映射。")
                continue # 跳过未映射的意图

            cypher_template = cypher_info[0]
            entity_type_needed = cypher_info[1] # "disease" 或 "drug" 或 "custom_query"

            current_entity_name = None
            placeholder_name = None

            if entity_type_needed == "disease":
                current_entity_name = disease_entity_name
                placeholder_name = "entity_name" # 修改为更通用的占位符
            elif entity_type_needed == "drug":
                current_entity_name = drug_entity_name
                placeholder_name = "entity_name" 
            elif entity_type_needed == "custom_food_query": # 特殊处理宜吃食物意图
                if disease_entity_name:
                    foods_do_eat = []
                    # 根据您提供的关系类型，只有 '疾病宜吃食物'，所以只查询这个
                    try:
                        res_do_eat = self.client.run("MATCH (d:疾病 {名称: $disease_name})-[:疾病宜吃食物]->(f:食物) RETURN f.名称 AS 名称", disease_name=disease_entity_name).data()
                        if res_do_eat: foods_do_eat = [item['名称'] for item in res_do_eat if item and item.get('名称')]
                    except Exception as e: 
                        print(f"Error querying 疾病宜吃食物 (do_eat) for {disease_entity_name}: {e}")
                        traceback.print_exc()

                    all_good_foods = list(set(foods_do_eat)) # 合并并去重，这里没有recommand_eat
                    if all_good_foods:
                        prompt_parts.append(f"<提示>关于“{disease_entity_name}”的“查询疾病宜吃食物”：{'、'.join(all_good_foods)}</提示>")
                        if intent not in yitu_matched_in_kg: yitu_matched_in_kg.append(intent)
                    else:
                        prompt_parts.append(f"<提示>知识图谱中未找到“{disease_entity_name}”的“查询疾病宜吃食物”相关信息。</提示>")
                else: # 如果没有疾病实体，也无法查询宜吃食物
                     prompt_parts.append(f"<提示>需要识别到相关疾病实体才能查询“{intent}”。</提示>")
                continue # 处理完特殊查询后跳到下一个意图

            if not current_entity_name:
                # 如果当前意图需要实体，但实体未识别，则跳过此意图的查询
                # prompt_parts.append(f"<提示>需要识别到相关实体（如疾病名称或药品名称）才能查询“{intent}”。</提示>")
                continue 

            # 执行通用Cypher查询
            if self.client:
                print(f"DEBUG_GEN_PROMPT_INTERNAL: Executing Cypher for intent '{intent}': {cypher_template} with params: {{'{placeholder_name}': '{current_entity_name}'}}")
                try:
                    res_data = self.client.run(cypher_template, **{placeholder_name: current_entity_name}).data()
                    print(f"DEBUG_GEN_PROMPT_INTERNAL: Cypher query result for '{intent}': {res_data}")

                    formatted_res = None
                    if "value" in cypher_template: # 属性查询 (例如简介、病因等)
                        formatted_res = single_value_formatter(res_data, 'value')
                    elif "名称" in cypher_template: # 关系查询 (例如症状、药品等)
                        formatted_res = list_formatter(res_data, '名称')
                    
                    if formatted_res:
                        prompt_parts.append(f"<提示>关于“{current_entity_name}”的“{intent}”：{formatted_res}</提示>")
                        if intent not in yitu_matched_in_kg:
                            yitu_matched_in_kg.append(intent)
                    else:
                        prompt_parts.append(f"<提示>知识图谱中未找到“{current_entity_name}”的“{intent}”相关信息。</提示>")
                except Exception as e:
                    detailed_error = traceback.format_exc()
                    prompt_parts.append(f"<提示>查询“{current_entity_name}”的“{intent}”信息时出错: {str(e)[:100]}...</提示>")
                    print(f"ERROR: Cypher query failed for intent '{intent}' with entity {current_entity_name}: {e}")
                    print(f"Failed Cypher Query Template: {cypher_template}")
                    print(f"Detailed Cypher Error: {detailed_error}")
                    # 如果是py2neo连接问题，尝试清除session_state中的client以便下一次尝试重连
                    if isinstance(e, py2neo.errors.ConnectionBroken) or isinstance(e, py2neo.errors.ServiceUnavailable):
                        st.session_state.neo4j_client = None 
            else:
                prompt_parts.append(f"<提示>知识图谱未连接，无法查询“{current_entity_name}”的“{intent}”。</提示>")
        
        final_prompt_str = "\n".join(prompt_parts)
        # 如果没有成功匹配到任何知识图谱信息，则返回通用心理支持提示
        if not final_prompt_str.strip() or not yitu_matched_in_kg : 
            final_prompt_str = "<提示>知识库中没有找到与您问题直接相关的具体医学信息。不过，别担心，我会尽力从心理支持的角度给您一些建议。</提示>"
        
        return final_prompt_str, "、".join(yitu_matched_in_kg) if yitu_matched_in_kg else "无知识图谱匹配信息", entities

    # 获取意图和知识，并组织中间结果
    def get_intention_and_knowledge(self, query: str):
        intention_chain = self.intention_prompt_template | self.intention_llm
        try:
            intention_ai_message = intention_chain.invoke({"query": query})
            intention_output_str = ""
            if hasattr(intention_ai_message, 'content'):
                intention_output_str = intention_ai_message.content
            elif isinstance(intention_ai_message, str):
                intention_output_str = intention_ai_message
            else:
                st.warning(f"未知的意图识别输出类型: {type(intention_ai_message)}")
                intention_output_str = str(intention_ai_message)
        except Exception as e:
            st.error(f"意图识别模型调用失败: {e}")
            traceback.print_exc()
            intention_output_str = '["查询疾病专家建议"] # 调用失败，使用默认值' 

        parsed_intentions = self._parse_intention_output(intention_output_str)
        
        try:
            entities = self._get_ner_results(query)
        except Exception as e:
            st.error(f"命名实体识别失败: {e}")
            traceback.print_exc()
            entities = {}
        
        knowledge_prompt_str, matched_yitu_str, final_entities = self._generate_knowledge_prompt(entities, parsed_intentions)
        
        return {
            "query": query,
            "raw_intention_output": intention_output_str,
            "parsed_intentions": parsed_intentions,
            "entities": final_entities, 
            "knowledge_prompt": knowledge_prompt_str, 
            "matched_yitu_str": matched_yitu_str 
        }

    # 流式生成答案
    def stream_answer(self, answer_llm, intermediate_results: dict, callbacks: list):
        answer_chain = self.answer_prompt_template | answer_llm
        full_response_content = ""
        
        current_query = intermediate_results["query"]
        current_knowledge_prompt = intermediate_results["knowledge_prompt"]
        
        print("-" * 50)
        print(f"DEBUG_STREAM_ANSWER: Query to LLM: {current_query}")
        print(f"DEBUG_STREAM_ANSWER: Knowledge prompt part for LLM:\n{current_knowledge_prompt}")
        try:
            final_prompt_for_llm = self.answer_prompt_template.format(query=current_query, prompt=current_knowledge_prompt)
            print(f"DEBUG_STREAM_ANSWER: Estimated full prompt length for LLM: {len(final_prompt_for_llm)} characters")
        except Exception as e:
            print(f"DEBUG_STREAM_ANSWER: Error formatting final prompt for length estimation: {e}")
        print("-" * 50)

        try:
            for chunk in answer_chain.stream(
                {"query": current_query, "prompt": current_knowledge_prompt},
                config={"callbacks": callbacks}
            ):
                if hasattr(chunk, 'content'):
                    full_response_content += chunk.content
                elif isinstance(chunk, str): 
                    full_response_content += chunk

            return full_response_content
        except Exception as e:
            st.error(f"答案生成流式调用失败: {e}")
            detailed_error = traceback.format_exc()
            print(f"答案流式生成过程中出错: {detailed_error}")
            try: 
                # 备选：如果流式失败，尝试非流式调用一次
                backup_response = answer_chain.invoke(
                    {"query": current_query, "prompt": current_knowledge_prompt}
                )
                if hasattr(backup_response, 'content'):
                    return backup_response.content
                return str(backup_response)
            except Exception as e_backup:
                st.error(f"备选答案生成调用也失败: {e_backup}")
                traceback.print_exc()
                return "抱歉，我在生成回答时遇到了连接问题，请稍后重试。"

# 缓存BERT模型组件，避免重复加载
@st.cache_resource
def load_bert_components():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__)) 
        with open(os.path.join(base_dir, 'tmp_data/tag2idx.npy'), 'rb') as f:
            tag2idx = pickle.load(f)
        idx2tag = list(tag2idx)

        if not hasattr(zwk, 'rule_find') or not hasattr(zwk, 'tfidf_alignment') or not hasattr(zwk, 'Bert_Model'):
            st.error("ner_model (zwk) 模块缺少必要的函数/类。请确保 `ner_model.py` 完整。")
            return None

        rule = zwk.rule_find()
        tfidf_r = zwk.tfidf_alignment()
        
        model_name_path = os.path.join(base_dir, 'model/chinese-roberta-wwm-ext')
        model_weights_path = os.path.join(base_dir, 'model/best_roberta_rnn_model_ent_aug.pt')

        if not os.path.exists(model_name_path):
            st.error(f"BERT模型路径未找到: {model_name_path}。请检查 `model` 文件夹。")
            return None
        if not os.path.exists(model_weights_path):
            st.error(f"BERT模型权重未找到: {model_weights_path}。请检查 `model` 文件夹。")
            return None

        bert_tokenizer = BertTokenizer.from_pretrained(model_name_path)
        bert_model = zwk.Bert_Model(model_name_path, hidden_size=128, tag_num=len(tag2idx), bi=True)
        bert_model.load_state_dict(torch.load(model_weights_path, map_location=device))
        bert_model = bert_model.to(device)
        bert_model.eval()
        print("成功: BERT NER组件已加载。")
        return {'tokenizer': bert_tokenizer, 'model': bert_model, 'idx2tag': idx2tag,
                'rule': rule, 'tfidf_r': tfidf_r, 'device': device}
    except FileNotFoundError as fnf_error:
        st.error(f"NER模型文件未找到: {fnf_error}。请检查路径。")
        return None
    except Exception as e:
        st.error(f"加载BERT NER组件失败: {e}")
        traceback.print_exc()
        return None

# 缓存Qwen意图识别模型
@st.cache_resource
def load_local_qwen_intention_llm():
    try:
        if not os.path.exists(LOCAL_INTENTION_MODEL_PATH):
            st.error(f"本地意图识别模型路径不存在: {LOCAL_INTENTION_MODEL_PATH}。请检查模型文件。")
            return None
        
        llm = HuggingFacePipeline.from_model_id(
            model_id=LOCAL_INTENTION_MODEL_PATH,
            task="text-generation",
            device_map="auto",  
            model_kwargs={"torch_dtype": torch.bfloat16, "trust_remote_code": True}, 
            # 移除 stop_sequences 参数，因为Qwen模型不支持
            pipeline_kwargs={"max_new_tokens": 100, "temperature": 0.1, "top_p": 0.9, "do_sample": True, "num_return_sequences": 1} 
        )
        
        st.success("本地意图识别模型 (Qwen) 加载成功！")
        return llm
    except Exception as e:
        print(f"load_local_qwen_intention_llm 中的详细错误: {traceback.format_exc()}")
        st.error(f"加载本地意图模型 (Qwen) 失败: {e}")
        return None

# 加载DeepSeek API模型
def load_deepseek_api_llm(api_model_identifier: str):
    api_key = DEEPSEEK_API_KEY_FROM_USER
    base_url = DEEPSEEK_API_BASE_URL
    if not api_key or api_key == "sk-YOUR_API_KEY_HERE": 
        st.error("DeepSeek API Key缺失或为占位符。请配置它。")
        return None
    if not base_url:
        st.error("DeepSeek API Base URL未配置。")
        return None
    try:
        llm = ChatOpenAI(
            model=api_model_identifier,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.6, 
            max_tokens=4000,
            request_timeout=300 
        )
        print(f"成功: DeepSeek API LLM ({api_model_identifier}) 已配置。")
        return llm
    except Exception as e:
        st.error(f"连接 DeepSeek API LLM ({api_model_identifier}) 失败: {e}")
        traceback.print_exc()
        return None

# 全局调试开关，由管理员在侧边栏控制
show_ent = False
show_int = False
show_prompt = False

# Streamlit 应用主函数
def main(is_admin, usname):
    global show_ent, show_int, show_prompt 

    st.set_page_config(page_title="心理咨询问答系统", layout="centered")
    st.title("心理咨询问答系统")

    # 侧边栏导航和功能
    with st.sidebar:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(base_dir, "img", "logo.jpg")
        if os.path.exists(logo_path):
            st.image(logo_path, use_column_width=True)
        else:
            st.info("Logo (img/logo.jpg) 未找到。")

        st.caption(
            f"""<p align="left">欢迎您，{'管理员' if is_admin else '用户'} {usname}！<br>当前版本：1.0</p>""", 
            unsafe_allow_html=True,
        )

        # 对话窗口管理
        if 'chat_windows' not in st.session_state:
            st.session_state.chat_windows = [[]]
            st.session_state.messages = [[]]
            st.session_state.active_window_index = 0

        if st.button('新建对话窗口', key="new_chat_window_sidebar_button_v5"): 
            st.session_state.chat_windows.append([])
            st.session_state.messages.append([])
            st.session_state.active_window_index = len(st.session_state.chat_windows) - 1
            st.rerun() 

        window_options = [f"对话窗口 {i + 1}" for i in range(len(st.session_state.chat_windows))]
        # 确保至少有一个对话窗口
        if not window_options: 
            st.session_state.chat_windows.append([])
            st.session_state.messages.append([])
            window_options = ["对话窗口 1"]
            st.session_state.active_window_index = 0

        selected_window_label = st.selectbox(
            '请选择对话窗口:', window_options,
            index=st.session_state.get('active_window_index', 0),
            key="selectbox_chat_window_sidebar_v5" 
        )
        st.session_state.active_window_index = window_options.index(selected_window_label)

        # 大模型选择
        selected_deepseek_model_name = st.selectbox(
            '请选择回答大模型:',
            list(DEEPSEEK_API_MODELS.keys()),
            key="selectbox_deepseek_model_sidebar_v5" 
        )
        # 根据选择更新当前使用的模型ID，并在模型切换时清除旧的缓存
        new_api_identifier = DEEPSEEK_API_MODELS[selected_deepseek_model_name]
        if st.session_state.get('current_deepseek_api_identifier') != new_api_identifier:
            st.session_state.current_deepseek_api_identifier = new_api_identifier
            if 'answer_llm' in st.session_state: del st.session_state['answer_llm']
            if 'active_answer_llm_id' in st.session_state: del st.session_state['active_answer_llm_id']
        
        # 管理员调试选项
        if is_admin:
            st.markdown("---")
            st.subheader("调试选项")
            show_ent = st.checkbox("显示实体识别结果", value=show_ent, key="cb_show_ent_sidebar_v5")
            show_int = st.checkbox("显示意图识别结果", value=show_int, key="cb_show_int_sidebar_v5")
            show_prompt = st.checkbox("显示知识图谱信息", value=show_prompt, key="cb_show_prompt_sidebar_v5")
            if st.button('修改知识图谱 (Neo4j Browser)', key="btn_edit_kg_sidebar_v5"):
                st.markdown(f'[点击打开 Neo4j Browser](http://localhost:7474/browser/)', unsafe_allow_html=True)
        else: # 非管理员默认关闭调试显示
            show_ent, show_int, show_prompt = False, False, False

        st.markdown("---")
        # 退出登录按钮
        if st.button("退出登录", key="btn_logout_sidebar_main_webui_unique_v5"): 
            keys_to_clear = [k for k in st.session_state.keys() if k not in ['is_admin_login', 'usname_login', 'logged_in']]
            for key in keys_to_clear:
                del st.session_state[key]
            st.session_state.logged_in = False 
            st.rerun() 

    active_window_index = st.session_state.active_window_index

    # 加载BERT组件（缓存）
    bert_components = load_bert_components()

    # Neo4j 客户端连接（缓存并处理重连）
    if 'neo4j_client' not in st.session_state or st.session_state.neo4j_client is None:
        try:
            st.session_state.neo4j_client = py2neo.Graph(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                name="neo4j" 
            )
            st.session_state.neo4j_client.run("RETURN 1") # 测试连接是否可用
            print("成功: 已连接到 Neo4j。")
            if 'kg_connection_error_shown' in st.session_state: # 如果之前显示过错误，连接成功后清除标记
                del st.session_state['kg_connection_error_shown']
        except (py2neo.errors.ServiceUnavailable, py2neo.errors.AuthError, Exception) as e: 
            if 'kg_connection_error_shown' not in st.session_state:
                st.error(f"知识图谱连接失败: {e}. 部分功能可能受限。请检查数据库是否正在运行、认证信息和网络通畅。")
                st.session_state.kg_connection_error_shown = True
            st.session_state.neo4j_client = None
    client = st.session_state.get('neo4j_client')

    # 加载Qwen意图识别模型（缓存）
    if 'intention_llm' not in st.session_state or st.session_state.intention_llm is None:
        st.session_state.intention_llm = load_local_qwen_intention_llm()
    intention_llm = st.session_state.get('intention_llm')

    # 初始化MedicalQAChain
    if 'qa_chain' not in st.session_state or st.session_state.qa_chain is None:
        if client and intention_llm and bert_components and bert_components.get('device'):
            st.session_state.qa_chain = MedicalQAChain(
                client, bert_components, bert_components['device'], intention_llm
            )
        else:
            st.session_state.qa_chain = None # 如果关键组件未加载，qa_chain将不可用
    qa_chain = st.session_state.get('qa_chain')

    # 加载DeepSeek答案生成模型（缓存）
    current_api_model_id_for_answer = st.session_state.get('current_deepseek_api_identifier', DEEPSEEK_API_MODELS[list(DEEPSEEK_API_MODELS.keys())[0]])
    if st.session_state.get('active_answer_llm_id') != current_api_model_id_for_answer or \
       'answer_llm' not in st.session_state or st.session_state.answer_llm is None:
        answer_llm_candidate = load_deepseek_api_llm(current_api_model_id_for_answer)
        st.session_state.answer_llm = answer_llm_candidate 
        st.session_state.active_answer_llm_id = current_api_model_id_for_answer if answer_llm_candidate else None
    answer_llm = st.session_state.get('answer_llm')

    # 显示聊天历史
    current_messages_history = st.session_state.messages[active_window_index]
    for msg_idx, message_data in enumerate(current_messages_history):
        with st.chat_message(message_data["role"]):
            st.markdown(message_data["content"])
            # 管理员模式下显示调试信息
            if message_data["role"] == "assistant":
                if show_ent and "entities" in message_data:
                    with st.expander(f"实体识别结果 (消息 {msg_idx+1})"): st.write(message_data["entities"])
                if show_int:
                    if "matched_yitu_str" in message_data:
                        with st.expander(f"意图识别 (知识图谱匹配) (消息 {msg_idx+1})"): st.write(message_data["matched_yitu_str"])
                    if "raw_intention_output" in message_data:
                        with st.expander(f"原始意图模型输出 (消息 {msg_idx+1})"): st.text(message_data["raw_intention_output"])
                    if "parsed_intentions" in message_data:
                        with st.expander(f"解析后意图列表 (消息 {msg_idx+1})"): st.write(message_data["parsed_intentions"])
                if show_prompt and "knowledge_prompt" in message_data:
                    with st.expander(f"知识图谱信息 (消息 {msg_idx+1})"): st.markdown(message_data["knowledge_prompt"])


    if query := st.chat_input("请输入您的问题...", key=f"chat_input_win_{active_window_index}_unique_v5"): 
        current_messages_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # 检查问答核心链和答案模型是否就绪
            if not qa_chain:
                st.error("问答核心链未就绪，请检查模型和知识图谱连接。")
                current_messages_history.append({"role": "assistant", "content": "系统组件未就绪，无法回答。"})
                return 

            if not answer_llm:
                st.error("回答生成大模型未就绪。请检查API Key和网络连接。")
                current_messages_history.append({"role": "assistant", "content": "回答模型未就绪，无法回答。"})
                return

            assistant_response_area = st.empty() # 用于流式输出
            stream_handler = StreamHandler(assistant_response_area)
            full_answer_content = ""
            try:
                # 获取意图、实体和知识图谱信息
                intermediate_results = qa_chain.get_intention_and_knowledge(query)
                # 流式生成答案
                full_answer_content = qa_chain.stream_answer(answer_llm, intermediate_results, callbacks=[stream_handler])
                
                # 将助手的完整回复及调试信息保存到历史记录
                assistant_message_data = {
                    "role": "assistant", "content": full_answer_content,
                    "matched_yitu_str": intermediate_results.get("matched_yitu_str", "N/A"),
                    "knowledge_prompt": intermediate_results.get("knowledge_prompt", "N/A"),
                    "entities": str(intermediate_results.get("entities", {})),
                    "raw_intention_output": intermediate_results.get("raw_intention_output", "N/A"),
                    "parsed_intentions": intermediate_results.get("parsed_intentions", [])
                }
                current_messages_history.append(assistant_message_data)

            except Exception as e:
                detailed_error = traceback.format_exc()
                st.error(f"处理您的请求时发生错误。")
                print(f"聊天处理过程中出错: {detailed_error}") 
                error_message_for_user = "抱歉，处理您的问题时出现了一个内部错误，这可能是网络连接或服务不稳定导致的。请稍后重试。"
                assistant_response_area.markdown(error_message_for_user)
                current_messages_history.append({"role": "assistant", "content": error_message_for_user, "error_details": detailed_error})

    st.session_state.messages[active_window_index] = current_messages_history

# 应用启动入口
if __name__ == "__main__":
    # 简单的登录状态模拟（可根据实际需求扩展为真正的登录系统）
    if 'logged_in' not in st.session_state: 
        st.session_state.logged_in = True 
        st.session_state.admin = True 
        st.session_state.usname = "测试管理员"

    if st.session_state.get('logged_in', False):
        main(is_admin=st.session_state.get('admin', False), usname=st.session_state.get('usname', "用户"))
    else:
        st.error("请先登录。")