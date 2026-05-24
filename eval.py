"""
RAGAS 评估：量化检索与生成质量
"""
import json
from rag_pipeline import get_pipeline
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# ── 测试问题（覆盖不同问题类型）─────────────────────────────────
TEST_QUESTIONS = [
    # 事实型
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
    # 定义型
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
    # 数值约束型
    {
        "question": "What is the maximum engine oil consumption allowed in the 2026 regulations?",
        "reference": "Engine oil consumption must never exceed 0.30 liters per 100 km."
    },
    {
        "question": "How many wastegates and pop-off valves are permitted on the power unit?",
        "reference": "The power unit may be equipped with a maximum of two wastegates and two pop-off valves."
    },
    # 规则限制型
    {
        "question": "Are variable geometry systems allowed in the turbocharger?",
        "reference": "Variable geometry systems are permitted in certain contexts as specified in Article 5.5."
    },
    {
        "question": "What safety equipment must be installed in a 2026 F1 car?",
        "reference": "Fire extinguishers, rear view mirrors, rear lights, safety tethers, safety harnesses, "
                    "driver cooling systems, and lateral safety lights are required."
    },
]


def run_evaluation():
    """运行 RAGAS 评估"""
    pipeline = get_pipeline()

    eval_data = []
    for item in TEST_QUESTIONS:
        result = pipeline.query(item["question"])
        # 合并检索到的上下文字段
        contexts = [s["preview"] for s in result["sources"]]
        eval_data.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": contexts,
            "ground_truth": item["reference"],
        })

    dataset = Dataset.from_list(eval_data)

    print("Running RAGAS evaluation...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n=== RAGAS Evaluation Results ===\n")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

    print(f"\n  Average Faithfulness:     {scores.get('faithfulness', 0):.4f}")
    print(f"  Average Answer Relevancy: {scores.get('answer_relevancy', 0):.4f}")
    print(f"  Average Context Precision: {scores.get('context_precision', 0):.4f}")
    print(f"  Average Context Recall:   {scores.get('context_recall', 0):.4f}")

    # 保存结果
    result_path = Path(__file__).parent / "eval_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "scores": {k: float(v) for k, v in scores.items()},
            "test_questions": TEST_QUESTIONS,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    from pathlib import Path
    run_evaluation()
