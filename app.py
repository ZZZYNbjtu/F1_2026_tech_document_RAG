"""
Gradio Web 界面：F1 2026 技术规则 RAG 问答（支持对话记忆）
"""
import gradio as gr
from rag_pipeline import get_pipeline


def answer_question(question: str, history: list[list[str]]) -> str:
    """处理用户问题，返回格式化回答"""
    if not question.strip():
        return "Please enter a question."

    # 将 Gradio 格式 [[user, bot], ...] 转为 Pipeline 格式
    formatted_history = []
    for user_msg, bot_msg in history:
        formatted_history.append(f"User: {user_msg}")
        formatted_history.append(f"Assistant: {bot_msg}")

    pipeline = get_pipeline()
    result = pipeline.query(question, history=formatted_history)

    # ── 拼接输出 ─────────────────────────────────────────
    output = ""

    # 如果问题被改写，先展示改写结果
    if result.get("standalone_query"):
        output += f"> *Rephrased for retrieval: {result['standalone_query']}*\n\n"

    # LLM 回答
    output += result["answer"]

    # 来源原文（可折叠，方便验证）
    output += "\n\n---\n### Sources (click to expand and verify)\n"
    for i, s in enumerate(result["sources"], 1):
        output += (
            f"<details>\n"
            f"<summary><b>[{i}] Article {s['article']}, Section {s['section']} "
            f"(score: {s['score']:.3f})</b></summary>\n\n"
            f"{s['content']}\n"
            f"</details>\n"
        )

    return output


def build_ui():
    get_pipeline()  # 启动时加载索引
    print("Pipeline loaded. Starting Gradio...")

    demo = gr.ChatInterface(
        fn=answer_question,
        title="F1 2026 Technical Regulations Q&A",
        description="Ask questions about the 2026 FIA Formula 1 Technical Regulations. "
                    "Conversation memory is supported — you can ask follow-up questions.",
        examples=[
            "What is the minimum weight of a 2026 F1 car?",
            "How much power can the MGU-K deliver?",
            "What are the engine specifications for 2026?",
            "What is RV-FLOOR-BODY?",
            "How many cylinders must the 2026 F1 engine have?",
            "What safety structures are required for the survival cell?",
            "What materials are prohibited in F1 2026?",
        ],
    )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, share=False)
