# F1 2026 Technical Regulations RAG

基于 FIA 官方 2026 F1 技术规则（207页）构建的 RAG 问答系统。

**技术栈**: 阿里百炼 text-embedding-v3 · ChromaDB · BM25 · DeepSeek · DashScope Reranker · Gradio

## 架构

```
用户问题 → 查询扩展（术语对齐）→ 混合检索（向量 + BM25 → RRF融合）→ Reranker精排 → DeepSeek生成 → 回答+来源引用
```

### 关键设计选择

**混合检索 + Reranker 精排**
向量检索（语义匹配）和 BM25（关键词匹配）各有盲区。先用两者宽召回 20 个候选，再用 DashScope gte-rerank 做交叉编码精排到 5 个。规则文档里既有精确术语（"MGU-K"、"5.3.2"）也有同义表述（"weight" vs "mass"），两级检索能互补。

**查询扩展（Query Expansion）**
用户提问的用词往往和文档术语不一致（"最低重量" vs "Minimum Mass"）。检索前用 LLM 将问题改写成富含文档术语的检索查询，提高召回质量。

**结构化分块 + 上下文锚点**
法规文档有严格的层次结构（Article → Section → Subsection）。短条款独立成块但前缀带上父级标题做上下文，避免 embedding 失去方向。

## 快速开始

### 1. 环境配置

```bash
conda create -n f1_rag python=3.11 -y
conda activate f1_rag
pip install -r requirements.txt
```

### 2. 设置 API Key

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

### 3. 运行

```bash
# Step 1: 解析 PDF + 分块
python parse_pdf.py

# Step 2: 构建索引（embedding 只跑一次，结果持久化到 chroma_db/）
python build_index.py

# Step 3: 启动 Web 界面
python app.py

# 可选: 运行评估
python eval.py
```

## 项目结构

```
f1_rag/
├── config.py          # 全局配置
├── parse_pdf.py       # PDF 解析 + 清洗 + 结构化分块
├── build_index.py     # 构建向量索引 + BM25 索引
├── rag_pipeline.py    # 混合检索 + RAG 生成核心
├── app.py             # Gradio Web 界面
├── eval.py            # 评估脚本（DeepSeek 打分）
├── data/              # 原始 PDF + 解析后数据
├── chroma_db/         # 向量库持久化
└── requirements.txt
```

## 评估结果

使用 DeepSeek 作为评估模型，在 11 道测试题上打分（0-10）：

| 指标 | 分数 | 说明 |
|------|------|------|
| Faithfulness | 8.2 /10 | 回答是否忠于检索到的上下文（不瞎编） |
| Answer Relevancy | 9.5 /10 | 回答是否直接切题 |
| Context Precision | 3.5 /10 | 检索到的文档块与问题的相关性 |

Context Precision 是当前主要优化方向。分析发现根因在分块策略——短条款被过度合并导致 embedding 稀释。后续可通过调整合并阈值或引入更细粒度的分块来提升。Faithfulness 和 Answer Relevancy 通过严格 Prompt + 查询扩展 + Reranker 已达到实用水平。

## License

本项目仅用于学习和展示目的。F1 Technical Regulations 版权归 FIA 所有。
