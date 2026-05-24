"""
RAG Pipeline：混合检索 + DeepSeek 生成
"""
import json
import re
import numpy as np
from pathlib import Path
from openai import OpenAI
import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from config import (
    PARSED_TEXT_PATH, CHROMA_DIR,
    DASHSCOPE_API_KEY, DEEPSEEK_API_KEY,
    EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE,
    LLM_MODEL, LLM_BASE_URL,
    VECTOR_TOP_K, BM25_TOP_K, FINAL_TOP_K,
)

# ── Prompt 模板 ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert on Formula 1 technical regulations, specializing in the 2026 FIA F1 Technical Regulations.
Answer the user's question based ONLY on the provided context passages.
If the context does not contain enough information to answer the question, say: "The 2026 regulations do not appear to cover this specific question."
When answering, cite the relevant article/section numbers from the context (e.g., "According to Article 5.3.2...").
Answer in the same language as the user's question (e.g. if the user asks in Chinese, answer in Chinese).
Keep answers concise but technically precise."""

QUERY_REWRITE_PROMPT = """Given the conversation history, rewrite the user's latest question into a standalone question that can be understood without prior context.
If the user says things like "answer again in Chinese" or "explain differently", preserve the original question's meaning but adjust the language/format request.
If the user simply asks a new question, return it unchanged.
Output ONLY the rewritten question, nothing else.

Conversation history:
{history}

Latest question: {question}

Rewritten question:"""


class HybridRetriever:
    """混合检索器：向量检索 + BM25 关键词检索，RRF 融合"""

    def __init__(self):
        self.chunks = self._load_chunks()
        self.chroma_collection = self._load_chroma()
        self.bm25, self.bm25_texts = self._load_bm25()
        print(f"Retriever ready: {len(self.chunks)} chunks")

    def _load_chunks(self) -> list[dict]:
        with open(PARSED_TEXT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_chroma(self):
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_collection("f1_regulations")

    def _load_bm25(self) -> tuple[BM25Okapi, list[str]]:
        bm25_path = CHROMA_DIR / "bm25_index.npz"
        if bm25_path.exists():
            data = np.load(bm25_path, allow_pickle=True)
            # 用 chunk 文本重建 BM25
            texts = [c["content"] for c in self.chunks]
            tokenized = [self._tokenize(t) for t in texts]
            bm25 = BM25Okapi(tokenized)
            return bm25, texts
        return None, []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'[a-zA-Z0-9]+', text.lower())

    def _embed_query(self, query: str) -> list[float]:
        """调用 DashScope 向量化查询"""
        import dashscope
        import time
        from dashscope import TextEmbedding
        dashscope.api_key = DASHSCOPE_API_KEY

        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v3,
            input=[query],
            dimension=EMBEDDING_DIM,
        )
        if resp.status_code == 200:
            return resp.output["embeddings"][0]["embedding"]
        else:
            time.sleep(1)
            resp = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v3,
                input=[query],
                dimension=EMBEDDING_DIM,
            )
            if resp.status_code == 200:
                return resp.output["embeddings"][0]["embedding"]
            raise RuntimeError(f"Query embedding failed: {resp.message}")

    def _vector_search(self, query_embedding: list[float], top_k: int) -> list[tuple[int, float]]:
        """向量检索，返回 [(chunk_index, score), ...]"""
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances"],
        )
        indices = []
        for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
            idx = int(doc_id.replace("chunk_", ""))
            # ChromaDB 用 cosine distance (0-2), 转换为相似度 (1-0)
            score = 1.0 - (distance / 2.0)  # 归一化到 0-1，越高越好
            indices.append((idx, score))
        return indices

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """BM25 关键词检索"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # 归一化
        max_score = scores.max() if scores.max() > 0 else 1
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i] / max_score)) for i in top_indices if scores[i] > 0]

    def retrieve(self, query: str) -> list[dict]:
        """混合检索主方法"""
        # 向量检索
        query_emb = self._embed_query(query)
        vector_results = self._vector_search(query_emb, VECTOR_TOP_K)

        # BM25 检索
        bm25_results = self._bm25_search(query, BM25_TOP_K)

        # RRF 融合
        fused = self._rrf_fuse(vector_results, bm25_results, k=60)

        # 返回 top_k 个 chunk
        top_chunks = []
        for idx, score in fused[:FINAL_TOP_K]:
            chunk = self.chunks[idx].copy()
            chunk["score"] = round(score, 4)
            top_chunks.append(chunk)

        return top_chunks

    def _rrf_fuse(self, vec_results: list[tuple[int, float]],
                  bm25_results: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion"""
        scores = {}
        for rank, (idx, _) in enumerate(vec_results):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results


class RAGPipeline:
    """RAG 问答管线"""

    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm_client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=DEEPSEEK_API_KEY,
        )

    def _build_context(self, chunks: list[dict]) -> str:
        """拼接检索到的文档块作为上下文"""
        parts = []
        for i, ch in enumerate(chunks, 1):
            source = f"[{i}] Article {ch.get('article','?')}, Section {ch.get('section','?')}"
            parts.append(f"{source}\n{ch['content']}")
        return "\n\n---\n\n".join(parts)

    def _rewrite_query(self, question: str, history: list[str]) -> str:
        """结合对话历史，将追问改写为独立问题"""
        if not history:
            return question

        # 只取最近 5 轮对话，防止历史过长
        recent = history[-10:]
        history_str = "\n".join(recent)

        rewrite_resp = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": QUERY_REWRITE_PROMPT.replace(
                "{history}", history_str
            ).replace("{question}", question)}],
            temperature=0.0,
            max_tokens=200,
        )
        rewritten = rewrite_resp.choices[0].message.content.strip()
        return rewritten

    def query(self, question: str, history: list[str] | None = None) -> dict:
        """执行 RAG 查询（支持对话记忆）"""
        # 第一步：改写问题
        if history:
            standalone = self._rewrite_query(question, history)
        else:
            standalone = question

        # 第二步：用改写后的问题做检索
        chunks = self.retriever.retrieve(standalone)
        context = self._build_context(chunks)

        # 第三步：构建消息（包含对话历史）
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for msg in history[-10:]:
                # history 是 ["User: xxx", "Assistant: yyy"] 格式
                if msg.startswith("User:"):
                    messages.append({"role": "user", "content": msg[5:].strip()})
                elif msg.startswith("Assistant:"):
                    messages.append({"role": "assistant", "content": msg[10:].strip()})
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

        # 第四步：调用 LLM 生成
        response = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1500,
        )

        answer = response.choices[0].message.content

        # 返回完整来源文本（用于验证）
        sources = [
            {"article": ch.get("article", "?"), "section": ch.get("section", "?"),
             "score": ch.get("score", 0),
             "content": ch["content"]}
            for ch in chunks
        ]

        return {
            "question": question,
            "standalone_query": standalone if standalone != question else None,
            "answer": answer,
            "sources": sources,
        }


# ── 单例 ────────────────────────────────────────────────────────
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
