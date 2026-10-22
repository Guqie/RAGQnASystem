# 基于知识图谱与RAG的心理咨询问答系统

> 本项目为本科/研究生毕业设计，构建了一个融合知识图谱检索增强生成（RAG）技术的医疗心理咨询智能问答系统。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 项目简介

本系统面向心理健康咨询场景。知识图谱和疾病文档是两套后台知识库。用户用自然语言提问后，咨询 Agent 按下面的循环处理：

1. **安全检查**：识别自杀、自伤等危机表述。命中后停止检索和用药讨论，直接给出援助热线
2. **感知**：用词典（以及可选的 BERT NER）识别疾病、症状、药品
3. **长上下文**：最近几轮保留原文，更早的轮次压成摘要；需要时再按原话检索历史
4. **工具调用**：模型决定是否调用图谱查询、文档检索或历史记忆。模型不可用时，本地调度同一组工具
5. **自然语言查询**：图谱问题编译成 `1 实体 属性` 或 `2 实体 关系 目标类型`，执行后把语句和结果一起返回
6. **流式回答**：以心理咨询师的口吻按片段输出。没有 API Key 时，把知识库返回分段整理出来

---

## 🏗️ 系统架构

```
用户输入
   ↓
安全检查 ──危机──→ 固定援助话术（不再查图谱、不讨论用药），仍按句流式输出
   ↓ 通过
实体感知 + 长上下文（近期原文 + 早期摘要）
   ↓
工具调用：query_knowledge_graph / search_documents / recall_memory
   ↓
自然语言编译为查询语句，并执行
   ├─ 知识图谱：1 实体 属性，或 2 实体 关系 目标类型
   └─ 疾病文档：BM25 检索简介、病因等原文
   ↓
知识库返回（查询语句、图谱结果、文档片段）
   ↓
咨询师流式回答（DeepSeek；无密钥时分段整理检索结果）
   ↓
写回记忆并落盘，界面展示推理过程
```

---

## ✨ 核心技术亮点

### 1. 三层融合 NER 实体识别

| 层次 | 方法 | 作用 |
|------|------|------|
| 深度学习层 | BERT + 双向RNN + CRF | 序列标注，识别上下文语义实体 |
| 规则匹配层 | Aho-Corasick 多模式匹配 | 基于词典的高效精确匹配 |
| 对齐校正层 | TF-IDF 余弦相似度 | 将识别结果对齐到标准知识图谱词条 |

三层结果融合，优先保留更长实体，有效提升召回率与准确率。

### 2. 知识图谱结构

图谱包含 **8 类实体节点**，覆盖医疗领域核心概念：

```
疾病 ──[疾病的症状]──→ 疾病症状
疾病 ──[疾病使用药品]──→ 药品
疾病 ──[疾病推荐药品]──→ 药品
疾病 ──[疾病宜吃食物]──→ 食物
疾病 ──[疾病忌吃食物]──→ 食物
疾病 ──[疾病所需检查]──→ 检查项目
疾病 ──[疾病所属科目]──→ 科目
疾病 ──[治疗的方法]──→ 治疗方法
疾病 ──[疾病并发疾病]──→ 疾病
药品 ←─[生产]────────── 药品商
```

### 3. 双路知识库

- **知识图谱**：自然语言先编译成和 `data/nl2cyhper_data-Copy1.txt` 相同的查询语句，再执行 Cypher。属性查询形如 `1 抑郁症 疾病病因`，关系查询形如 `2 抑郁症 疾病使用药品 药品`。从症状反查疾病时写成 `2 失眠 疾病的症状 疾病`
- **疾病文档**：简介、病因等原文按字二元组做 BM25。完整语料在第一次检索时建立索引，并缓存到 `tmp_data/rag_index.pkl`
- 只有症状、没有疾病名时，先按共现推断，再对排名第一的疾病做第二跳，并写明「不是诊断」
- 查询语句、图谱命中和文档片段都出现在「Agent 推理过程」里，作为这一轮的知识库返回

### 4. 对话能力

- **流式回复**：模型输出按 token 送到界面；没有密钥时，整理后的回复也按句分段流出
- **长上下文**：最近 6 轮保留原文，更早的轮次压成摘要，历史最多保留 30 轮
- **历史记忆**：问到「上次 / 之前」时，用 `recall_memory` 在窗口之外的原话里检索
- **工具调用**：`query_knowledge_graph`、`search_documents`、`recall_memory`。工具接口失败时，改为本地执行同一组查询，再继续流式回答
- 对话窗口按用户名写到 `tmp_data/memory/`，刷新后还在
- 没有配置 `DEEPSEEK_API_KEY` 时，检索和整理回复仍然能完成一轮咨询

### 5. 回答模型

- DeepSeek 负责工具调用和最终回答，回答走流式接口
- 本地 Qwen 不再是启动系统的前置条件

---

## 🗂️ 项目结构

```
RAGQnASystem/
├── webui.py                  # Streamlit 入口，展示对话和 Agent 推理过程
├── login.py                  # 用户登录注册模块
├── agent/                    # 咨询 Agent
│   ├── core.py               # 安全检查、工具调度、流式回答
│   ├── nl2query.py           # 自然语言编译为图谱查询语句并执行
│   ├── rag.py                # 疾病文档 BM25 检索
│   ├── knowledge.py          # Neo4j / 内存图谱查询
│   ├── dictionary.py         # 词典实体识别
│   ├── memory.py             # 长上下文、历史检索、会话落盘
│   ├── llm.py                # DeepSeek 工具调用与流式输出
│   ├── safety.py             # 危机表述识别
│   └── demo.py               # 不连数据库的离线演示
├── ner_model.py              # NER模型定义、训练与推理
├── ner_data.py               # NER训练数据处理
├── user_data_storage.py      # 用户数据持久化
├── requirements.txt          # Python依赖列表
│
├── data/
│   ├── medical.json          # 原始医疗知识数据
│   ├── medical_new_2.json    # 处理后的医疗数据
│   ├── ner_data_aug.txt      # NER训练数据（数据增强后）
│   ├── build_up_graph.py     # 知识图谱构建脚本
│   ├── processjson.py        # 数据预处理脚本
│   ├── ent_aug/              # 实体词典（8类）
│   │   ├── 疾病.txt
│   │   ├── 疾病症状.txt
│   │   ├── 药品.txt
│   │   ├── 药品商.txt
│   │   ├── 食物.txt
│   │   ├── 检查项目.txt
│   │   ├── 治疗方法.txt
│   │   └── 科目.txt
│   └── lora_data/            # LoRA微调数据
│
├── model/
│   ├── chinese-roberta-wwm-ext/   # BERT预训练模型（需自行下载）
│   └── best_roberta_rnn_model_ent_aug.pt  # 训练好的NER模型权重
│
├── tmp_data/
│   └── tag2idx.npy           # NER标签映射文件
│
└── img/                      # 项目截图与说明图片
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Neo4j 5.x（本地或远程）
- CUDA（可选，有 GPU 推理更快）

### 1. 克隆项目

```bash
git clone https://github.com/Guqie/RAGQnASystem.git
cd RAGQnASystem
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，至少填入 Neo4j 密码。DeepSeek 密钥可以后补，没有它也能跑通检索和离线回复。

```bash
DEEPSEEK_API_KEY=your_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### 4. 下载 NER 模型（可选）

没有 BERT 权重时，系统使用 `data/ent_aug` 词典做实体识别。需要恢复深度学习 NER 时再下载 chinese-roberta-wwm-ext，并把训练权重放到 `model/best_roberta_rnn_model_ent_aug.pt`。

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('hfl/chinese-roberta-wwm-ext', cache_dir='./model')"
```

### 5. 配置 Neo4j 知识图谱

```bash
# 启动 Neo4j 后，运行图谱构建脚本
python data/build_up_graph.py
```

连接信息从环境变量读取，见上面的 `.env`。疾病属性查询同时兼容建图脚本里的「疾病简介 / 疾病病因 / 疾病易感人群」和旧字段名。

### 6. 训练 NER 模型（可选）

如果需要重新训练：

```bash
python ner_model.py
# 训练完成后权重保存至 model/best_roberta_rnn_model_ent_aug.pt
```

### 7. 先看离线 Agent 演示

不启动 Neo4j、也不配置 API Key：

```bash
python -m agent
```

演示会走三轮对话：疾病识别、追问用药、危机表述。终端会打印编译出的查询语句、图谱返回、文档片段和分段回复。知识库是内置样例，不是完整医疗库。

### 8. 启动系统

```bash
streamlit run login.py
```

浏览器访问 `http://localhost:8501`。默认管理员用户名是 `admin`。侧边栏可以展开每一轮的推理过程，里面有查询语句和两路检索结果。勾选「使用内置演示图谱」后，不连接 Neo4j 也会使用对应的演示文档。对话会记在本机 `tmp_data/memory/`。

---

## 💡 使用示例

**示例问题：**

```
我最近情绪低落，总是失眠，这是抑郁症吗？
我朋友得了焦虑症，应该怎么治疗？
抑郁症一般要多久才能好？治疗效果怎么样？
得了强迫症会引发其他精神问题吗？
```

**系统处理流程（以「焦虑症怎么治疗」为例）：**

```
输入：我朋友得了焦虑症，应该怎么治疗？

① 安全检查：通过
② 实体识别：疾病 = 焦虑症
③ 规划：query_knowledge，意图为治疗方法和常用药品
④ 图谱查询：治疗方法、常用药品
⑤ 生成回答：以心理咨询师口吻转述资料，并说明这不是诊断
```

追问「那要吃药吗」时，Agent 会从记忆里取回「焦虑症」，而不是把这句话当成一个新话题。如果用户出现自杀、自伤表述，循环会在安全检查处停下。

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| 前端框架 | Streamlit |
| NER 模型 | BERT (chinese-roberta-wwm-ext) + 双向RNN |
| 工具调用与答案生成 | DeepSeek API；无密钥时本地调用同一组工具并分段输出 |
| 多轮记忆 | 近期原文、早期摘要、历史检索，并按用户落盘 |
| 安全策略 | 危机表述短路，返回援助热线 |
| 知识图谱 | Neo4j 5.x |
| 图谱查询 | 自然语言语句编译为 Cypher |
| 文档检索 | 字二元组 BM25 |
| 大模型接口 | DeepSeek OpenAI 兼容接口 |
| NER 规则 | Aho-Corasick (pyahocorasick) |
| 实体对齐 | TF-IDF + 余弦相似度 (scikit-learn) |
| 深度学习 | PyTorch + HuggingFace Transformers |

---

## 📊 NER 模型性能

| 指标 | 数值 |
|------|------|
| 实体类别数 | 8 类 |
| 数据增强策略 | 实体替换、实体掩盖、实体拼接 |
| 推理设备 | CUDA / CPU 自适应 |

---

## ⚠️ 注意事项

1. `model/chinese-roberta-wwm-ext/` 和 `model/best_roberta_rnn_model_ent_aug.pt` 体积较大，已加入 `.gitignore`，需自行下载或训练
2. `tmp_data/user_credentials.json` 包含用户数据，已加入 `.gitignore`，不随代码上传
3. 请勿将 API Key 写进源码。使用项目根目录的 `.env`，该文件已被 git 忽略
4. 如果以前的提交里曾经包含过 API Key 或数据库密码，请在对应控制台作废旧密钥，只保留环境变量里的新密钥

---

## 🤝 致谢

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Neo4j](https://neo4j.com/)
- [DeepSeek](https://www.deepseek.com/)
- [Qwen](https://github.com/QwenLM/Qwen)

---

## 📄 License

MIT License
