# 基于知识图谱与RAG的心理咨询问答系统

> 本项目为本科/研究生毕业设计，构建了一个融合知识图谱检索增强生成（RAG）技术的医疗心理咨询智能问答系统。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 项目简介

本系统面向心理健康咨询场景。用户输入自然语言后，咨询 Agent 按下面的循环处理，而不是每次都走同一条固定流水线：

1. **安全检查**：识别自杀、自伤等危机表述。命中后停止检索和用药讨论，直接给出援助热线
2. **感知**：用词典（以及可选的 BERT NER）识别疾病、症状、药品
3. **记忆**：把本轮问题和上一轮关注的疾病接上，处理「那要吃药吗」这类追问
4. **规划**：由大模型选择工具；模型不可用或输出不合法时，退回关键词规则
5. **工具调用**：按计划查询 Neo4j，必要时先用症状共现推断疾病，再查第二跳
6. **回答**：以心理咨询师的口吻组织回复。没有 API Key 时，直接整理检索结果

---

## 🏗️ 系统架构

```
用户输入
   ↓
安全检查 ──危机──→ 固定援助话术（不再查图谱、不讨论用药）
   ↓ 通过
实体感知 + 对话记忆
   ↓
规划器（大模型 JSON 计划，失败则规则规划）
   ↓
工具：症状推断 / 疾病查询 / 药品生产商 / 直接咨询
   ↓
必要时第二跳：用推断出的疾病再查资料
   ↓
咨询师回答（DeepSeek 流式；无密钥时整理检索结果）
   ↓
写回记忆，并在界面展示推理过程
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

### 3. RAG 检索增强生成

- 根据识别出的意图动态生成 Cypher 查询语句
- 支持**症状反向推断疾病**（当用户描述症状而非疾病名时）
- 知识图谱信息作为上下文注入 Prompt，引导大模型生成有据可查的回答

### 4. 可解释的 Agent 循环

- 规划器只允许调用四类动作：`infer_diseases`、`query_knowledge`、`query_drug`、`counsel_only`
- 症状没有对应疾病名时，先做共现推断，再查询排名第一的疾病，并明确写上「不是诊断」
- 每一轮的安全检查、实体、计划和工具结果会显示在「Agent 推理过程」里
- 没有配置 `DEEPSEEK_API_KEY` 时，规则规划加检索结果仍然能完成一轮咨询

### 5. 回答模型

- DeepSeek 同时负责规划和最终回答，回答支持流式输出
- 本地 Qwen 不再是启动系统的前置条件

---

## 🗂️ 项目结构

```
RAGQnASystem/
├── webui.py                  # Streamlit 入口，展示对话和 Agent 推理过程
├── login.py                  # 用户登录注册模块
├── agent/                    # 咨询 Agent
│   ├── core.py               # 安全检查、规划、工具执行、回答
│   ├── planner.py            # 大模型规划与规则回退
│   ├── knowledge.py          # Neo4j / 内存图谱查询
│   ├── dictionary.py         # 词典实体识别
│   ├── memory.py             # 多轮记忆
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

演示会走三轮对话：疾病识别、追问用药、危机表述。图谱是内置样例，用来展示循环，不是完整医疗库。

### 8. 启动系统

```bash
streamlit run login.py
```

浏览器访问 `http://localhost:8501`。默认管理员用户名是 `admin`。侧边栏可以展开每一轮的推理过程；勾选「使用内置演示图谱」后，不连接 Neo4j 也能走完疾病识别、追问和危机检查。

---

## 💡 使用示例

**示例问题：**

```
我最近情绪很低落，总是提不起精神，这是抑郁症吗？
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
| 规划与答案生成 | DeepSeek API；无密钥时使用规则规划 |
| 多轮记忆 | 会话内疾病、症状和近期危机标记 |
| 安全策略 | 危机表述短路，返回援助热线 |
| 知识图谱 | Neo4j 5.x |
| 图谱查询 | Cypher |
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
