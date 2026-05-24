"""
构建索引：向量索引 (ChromaDB) + 关键词索引 (BM25)
"""
import json
import time
import numpy as np
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi
import dashscope
from dashscope import TextEmbedding

from config import (
    PARSED_TEXT_PATH, CHROMA_DIR,
    DASHSCOPE_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE,
)

dashscope.api_key = DASHSCOPE_API_KEY


def load_chunks() -> list[dict]:
    with open(PARSED_TEXT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def simple_tokenize(text: str) -> list[str]:
    """简单英文分词"""
    import re
    return re.findall(r'[a-zA-Z0-9]+', text.lower())


def generate_embeddings(texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    """
    调用阿里百炼 text-embedding-v3 生成向量
    文档: https://help.aliyun.com/zh/model-studio/text-embedding-api
    """
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} "
              f"({i + 1}-{min(i + batch_size, total)}/{total})")

        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v3,
            input=batch,
            dimension=EMBEDDING_DIM,
        )

        if resp.status_code == 200:
            embeddings = [item["embedding"] for item in resp.output["embeddings"]]
            all_embeddings.extend(embeddings)
        else:
            print(f"  Error: {resp.status_code} - {resp.message}")
            # 重试一次
            time.sleep(2)
            resp = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v3,
                input=batch,
                dimension=EMBEDDING_DIM,
            )
            if resp.status_code == 200:
                embeddings = [item["embedding"] for item in resp.output["embeddings"]]
                all_embeddings.extend(embeddings)
            else:
                raise RuntimeError(f"Embedding failed after retry: {resp.message}")

        time.sleep(0.3)  # 避免触发限流

    return all_embeddings


def build_chroma(chunks: list[dict], embeddings: list[list[float]]):
    """构建 ChromaDB 向量索引"""
    print("  Building ChromaDB index...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 如果已有 collection 就删掉重建
    try:
        client.delete_collection("f1_regulations")
    except Exception:
        pass

    collection = client.create_collection(
        name="f1_regulations",
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c["content"] for c in chunks]
    metadatas = [
        {"article": c.get("article", ""), "section": c.get("section", ""),
         "title": c.get("title", "")[:200]}
        for c in chunks
    ]

    # 分批写入 ChromaDB
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )

    print(f"  ChromaDB: {collection.count()} documents indexed")
    return collection


def build_bm25(chunks: list[dict]) -> tuple[BM25Okapi, list[str]]:
    """构建 BM25 关键词索引"""
    print("  Building BM25 index...")
    tokenized = [simple_tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"  BM25: {len(tokenized)} documents indexed")
    return bm25, [c["content"] for c in chunks]


def build_all():
    """主入口"""
    print("[1/3] Loading chunks...")
    chunks = load_chunks()
    print(f"      {len(chunks)} chunks loaded")

    # 生成 embedding
    print("[2/3] Generating embeddings...")
    texts = [c["content"] for c in chunks]
    embeddings = generate_embeddings(texts)
    print(f"      {len(embeddings)} embeddings generated")

    # 构建 ChromaDB
    print("[3/3] Building indexes...")
    build_chroma(chunks, embeddings)
    bm25_index, _ = build_bm25(chunks)

    # 保存 BM25 相关信息（用 numpy）
    bm25_path = CHROMA_DIR / "bm25_index.npz"
    #  BM25 的参数可以保存
    np.savez(
        bm25_path,
        doc_freqs=np.array(bm25_index.doc_freqs, dtype=object),
        doc_len=np.array(bm25_index.doc_len),
        avgdl=bm25_index.avgdl,
        corpus_size=bm25_index.corpus_size,
    )
    print(f"  BM25 metadata saved to {bm25_path}")
    print("\n=== Index build complete ===")


if __name__ == "__main__":
    build_all()
