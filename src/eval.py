"""
评估脚本：用 DeepSeek 量化 RAG 检索与生成质量
替代 RAGAS，避免依赖冲突，逻辑透明可解释
"""
import json
from pathlib import Path
from openai import OpenAI
from .rag_pipeline import get_pipeline
from .config import DEEPSEEK_API_KEY, LLM_MODEL, LLM_BASE_URL

client = OpenAI(base_url=LLM_BASE_URL, api_key=DEEPSEEK_API_KEY)

# ── 评估 Prompt ──────────────────────────────────────────────────
FAITHFULNESS_PROMPT = """Score how faithful the answer is to the given context (0-10).
A faithful answer makes NO claims beyond what the context supports.
10 = every claim is directly supported by the context
0 = the answer contradicts or fabricates information not in the context

Context:
{context}

Answer:
{answer}

Reply with ONLY a number (0-10) and a one-sentence reason. Format: "X/10 - reason"
"""

RELEVANCY_PROMPT = """Score how relevant and complete the answer is for the question (0-10).
10 = the answer fully and directly addresses the question
0 = the answer is completely off-topic or irrelevant

Question: {question}

Answer: {answer}

Reply with ONLY a number (0-10) and a one-sentence reason. Format: "X/10 - reason"
"""

CONTEXT_PRECISION_PROMPT = """Score how relevant the retrieved chunk is to the question (0-10).
10 = the chunk directly answers or contains key information for the question
0 = the chunk is completely irrelevant to the question

Question: {question}

Chunk:
{chunk}

Reply with ONLY a number (0-10). Format: "X/10"
"""

# ── 测试问题 ─────────────────────────────────────────────────────
TEST_QUESTIONS = [
    {
        "question": "What is the minimum weight of a 2026 F1 car excluding fuel?",
        "reference": "The minimum mass of the car excluding fuel must not be less than 795 kg."
    },
    {
        "question": "What is the engine cubic capacity requirement for the 2026 F1 power unit?",
        "reference": "Engine cubic capacity must be 1600cc (+0/-10cc)."
    },
    {
        "question": "How many cylinders must the 2026 F1 engine have and in what configuration?",
        "reference": "All engines must have six cylinders arranged in a 90 degree V configuration."
    },
    {
        "question": "How many intake and exhaust valves per cylinder are permitted?",
        "reference": "Engines must have two intake and two exhaust valves per cylinder."
    },
    {
        "question": "What is the maximum MGU-K power output?",
        "reference": "The MGU-K maximum power is 350 kW."
    },
    {
        "question": "What is the definition of the Power Unit according to Article 5.1.2?",
        "reference": "The power unit is the internal combustion engine and turbocharger with ancillaries, "
                    "the energy recovery system and all actuation systems and PU-Control electronics."
    },
    {
        "question": "What does ERS stand for and what is its purpose?",
        "reference": "ERS stands for Energy Recovery System, designed to recover energy from the car, "
                    "store that energy and make it available to propel the car."
    },
    {
        "question": "What is the maximum engine oil consumption allowed in the 2026 regulations?",
        "reference": "Engine oil consumption must never exceed 0.30 liters per 100 km."
    },
    {
        "question": "How many wastegates and pop-off valves are permitted on the power unit?",
        "reference": "The power unit may be equipped with a maximum of two wastegates and two pop-off valves."
    },
    {
        "question": "What safety equipment must be installed in a 2026 F1 car?",
        "reference": "Fire extinguishers, rear view mirrors, rear lights, safety tethers, safety harnesses, "
                    "driver cooling systems, and lateral safety lights are required."
    },
    {
        "question": "What is the 2026 F1 engine configuration (cylinders, layout, capacity)?",
        "reference": "The 2026 F1 engine is a 1600cc V6 with a 90-degree bank angle."
    },
]


def llm_score(prompt: str) -> tuple[int, str]:
    """调用 DeepSeek 打分，返回 (分数, 原因)"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    text = resp.choices[0].message.content.strip()
    try:
        score_str = text.split("/")[0].strip()
        score = int(float(score_str))
        reason = text.split("-", 1)[1].strip() if "-" in text else ""
    except (ValueError, IndexError):
        score = 0
        reason = f"parse error: {text[:80]}"
    return min(max(score, 0), 10), reason


def run_evaluation():
    pipeline = get_pipeline()
    results = []

    print(f"{'='*60}")
    print(f"Evaluating {len(TEST_QUESTIONS)} questions...")
    print(f"{'='*60}\n")

    for i, item in enumerate(TEST_QUESTIONS, 1):
        question = item["question"]
        print(f"[{i}/{len(TEST_QUESTIONS)}] {question[:80]}...")

        result = pipeline.query(question)
        answer = result["answer"]
        contexts = [s["content"] for s in result["sources"]]

        # 1. Faithfulness：回答是否忠于上下文
        faith_score, faith_reason = llm_score(
            FAITHFULNESS_PROMPT.format(
                context="\n---\n".join(contexts[:3]),
                answer=answer,
            )
        )

        # 2. Answer Relevancy：回答是否切题
        relevancy_score, relevancy_reason = llm_score(
            RELEVANCY_PROMPT.format(question=question, answer=answer)
        )

        # 3. Context Precision：检索到的每个 chunk 与问题的相关性
        chunk_scores = []
        for ch in result["sources"]:
            cs, _ = llm_score(
                CONTEXT_PRECISION_PROMPT.format(question=question, chunk=ch["content"][:800])
            )
            chunk_scores.append(cs)
        avg_precision = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0

        eval_result = {
            "question": question,
            "answer": answer[:500],
            "faithfulness": faith_score,
            "faithfulness_reason": faith_reason,
            "answer_relevancy": relevancy_score,
            "relevancy_reason": relevancy_reason,
            "context_precision": round(avg_precision, 1),
            "individual_chunk_scores": chunk_scores,
            "num_chunks_retrieved": len(result["sources"]),
            "top_source": f"Article {result['sources'][0]['article']}, "
                          f"Section {result['sources'][0]['section']}"
                          if result["sources"] else "N/A",
        }
        results.append(eval_result)

        print(f"  Faithfulness: {faith_score}/10 | Relevancy: {relevancy_score}/10 | "
              f"Context Precision: {avg_precision:.1f}/10")
        print(f"    → {faith_reason[:100]}\n")

    # ── 汇总 ─────────────────────────────────────────────────────
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_relev = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_prec = sum(r["context_precision"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"SUMMARY (averaged over {len(results)} questions)")
    print(f"{'='*60}")
    print(f"  Faithfulness:       {avg_faith:.1f}/10")
    print(f"  Answer Relevancy:   {avg_relev:.1f}/10")
    print(f"  Context Precision:  {avg_prec:.1f}/10")

    # 保存
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "avg_faithfulness": round(avg_faith, 1),
                "avg_answer_relevancy": round(avg_relev, 1),
                "avg_context_precision": round(avg_prec, 1),
                "num_questions": len(results),
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    run_evaluation()
