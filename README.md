# 基于知识图谱与RAG的心理咨询问答系统

> 本项目为本科/研究生毕业设计，构建了一个融合知识图谱检索增强生成（RAG）技术的医疗心理咨询智能问答系统。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 项目简介

本系统面向心理健康咨询场景，用户输入自然语言问题后，系统自动完成：

1. **命名实体识别（NER）**：识别问题中的疾病、药品、症状等医疗实体
2. **查询意图识别**：理解用户想查询的信息类型（病因、治疗方法、用药建议等）
3. **知识图谱检索**：从 Neo4j 医疗知识图谱中精准获取结构化信息
4. **大模型答案生成**：以心理咨询师的口吻，将知识图谱信息融合生成专业、有温度的回答

---

## 🏗️ 系统架构

```
用户输入自然语言问题
        ↓
┌─────────────────────────────────────┐
│         NER 实体识别模块             │
│  BERT+RNN 模型 + 规则匹配 + TF-IDF  │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│         意图识别模块                 │
│     Qwen2.5-3B-Instruct (本地)      │
│     16种预定义查询意图分类           │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│       知识图谱检索模块               │
│    Neo4j + Cypher 结构化查询        │
│    医疗实体关系图谱（8类实体）       │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│         答案生成模块                 │
│   DeepSeek API（流式输出）          │
│   心理咨询师角色 Prompt 设计        │
└─────────────────┬───────────────────┘
                  ↓
        Streamlit Web 界面展示
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

### 4. 双模型协作

- **本地模型**（Qwen2.5-3B）负责意图识别，低延迟、无需联网
- **API 模型**（DeepSeek）负责最终答案生成，效果更优、支持流式输出

---

## 🗂️ 项目结构

```
RAGQnASystem/
├── webui.py                  # 主程序入口，Streamlit Web界面 + 问答核心链
├── ner_model.py              # NER模型定义、训练与推理
├── ner_data.py               # NER训练数据处理
├── login.py                  # 用户登录注册模块
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
pip install langchain langchain-community langchain-openai
```

### 3. 下载模型

**BERT 预训练模型**（chinese-roberta-wwm-ext）：

```bash
# 方式一：ModelScope（国内推荐）
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('hfl/chinese-roberta-wwm-ext', cache_dir='./model')"

# 方式二：HuggingFace
git clone https://huggingface.co/hfl/chinese-roberta-wwm-ext model/chinese-roberta-wwm-ext
```

**Qwen2.5-3B-Instruct 意图识别模型**：

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct')"
```

> 下载后请确认 `webui.py` 中 `LOCAL_INTENTION_MODEL_PATH` 路径与实际路径一致。

### 4. 配置 Neo4j 知识图谱

```bash
# 启动 Neo4j 后，运行图谱构建脚本
python data/build_up_graph.py
```

修改 `webui.py` 中的连接信息：

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
```

### 5. 配置 DeepSeek API Key

推荐使用环境变量管理，避免硬编码：

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

然后在 `webui.py` 中改为：

```python
import os
DEEPSEEK_API_KEY_FROM_USER = os.environ.get("DEEPSEEK_API_KEY")
```

### 6. 训练 NER 模型（可选）

如果需要重新训练：

```bash
python ner_model.py
# 训练完成后权重保存至 model/best_roberta_rnn_model_ent_aug.pt
```

### 7. 启动系统

```bash
streamlit run webui.py
```

浏览器访问 `http://localhost:8501` 即可使用。

---

## 💡 使用示例

**示例问题：**

```
我最近情绪很低落，总是提不起精神，这是抑郁症吗？
我朋友得了焦虑症，应该怎么治疗？
抑郁症一般要多久才能好？治疗效果怎么样？
得了强迫症会引发其他精神问题吗？
```

**系统处理流程（以"焦虑症怎么治疗"为例）：**

```
输入：我朋友得了焦虑症，应该怎么治疗？

① NER识别：{ "疾病": "焦虑症" }
② 意图识别：["查询疾病的治疗方法", "查询疾病常用药品"]
③ KG查询：
   - 治疗方法：心理治疗、药物治疗、认知行为疗法...
   - 常用药品：阿普唑仑、劳拉西泮...
④ 生成回答：以心理咨询师口吻，融合知识图谱信息给出专业建议
```

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| 前端框架 | Streamlit |
| NER 模型 | BERT (chinese-roberta-wwm-ext) + 双向RNN |
| 意图识别 | Qwen2.5-3B-Instruct |
| 答案生成 | DeepSeek API (deepseek-chat / deepseek-reasoner) |
| 知识图谱 | Neo4j 5.x |
| 图谱查询 | Cypher |
| LLM 框架 | LangChain |
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
3. 请勿将 API Key 硬编码提交到公开仓库，建议使用环境变量管理

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
