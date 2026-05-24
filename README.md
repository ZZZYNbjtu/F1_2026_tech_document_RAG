# F1 2026 Technical Regulations RAG

基于 FIA 官方 2026 F1 技术规则（207页）构建的 RAG 问答系统。

**技术栈**: 阿里百炼 text-embedding-v3 · ChromaDB · BM25 · DeepSeek · Gradio · RAGAS

## 架构

```
用户问题 → 混合检索（向量 + BM25 → RRF融合）→ DeepSeek生成 → 回答+来源引用
```

### 为什么混合检索？

F1 技术规则中包含大量精确术语（如 RV-FLOOR-BODY, MGU-K）和编号（如 5.3.2）。纯向量检索对这类精确关键词匹配不够敏感，BM25 可以弥补这一点。RRF 融合让两者互补。

### 为什么结构化分块？

法规文档有严格的层次结构（Article → Section → Subsection）。按照这些边界切分能保持语义完整性，避免把 "3.5.1 的定义" 和 "3.5.2 的限制" 混在一起。

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

# 可选: 运行 RAGAS 评估
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
├── eval.py            # RAGAS 评估
├── data/              # 原始 PDF + 解析后数据
├── chroma_db/         # 向量库持久化
└── requirements.txt
```

## 评估结果

使用 RAGAS 在 11 道测试题上评估：

| 指标 | 分数 | 说明 |
|------|------|------|
| Faithfulness | - | 回答是否忠于上下文 |
| Answer Relevancy | - | 回答是否切题 |
| Context Precision | - | 检索到的文档是否相关 |
| Context Recall | - | 相关文档是否被检索到 |

## License

本项目仅用于学习和展示目的。F1 Technical Regulations 版权归 FIA 所有。
