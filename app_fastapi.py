"""
FastAPI 后端：REST API + SSE 流式输出
http://127.0.0.1:8000     → 聊天界面
http://127.0.0.1:8000/docs → Swagger 文档
"""
import json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_pipeline import get_pipeline, SYSTEM_PROMPT

app = FastAPI(title="F1 2026 Tech Regs RAG", version="2.0")

pipeline = get_pipeline()


class QueryRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    article: str
    section: str
    score: float
    content: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """单轮问答：提交问题，返回完整回答 + 来源"""
    result = pipeline.query(req.question)
    return result


@app.post("/chat")
async def chat(req: QueryRequest):
    """多轮对话：SSE 流式输出，逐 token 推送"""
    import asyncio

    # 先检索
    result = pipeline.query(req.question)
    chunks = result["sources"]
    context = pipeline._build_context(chunks)

    # 流式生成
    response = pipeline.llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"},
        ],
        temperature=0.1,
        max_tokens=1500,
        stream=True,
    )

    async def generate():
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield f"data: {json.dumps({'content': delta.content})}\n\n"
        # 最后发送来源
        sources_data = [
            {"article": s.get("article", "?"), "section": s.get("section", "?"),
             "score": s.get("score", 0)}
            for s in chunks
        ]
        yield f"data: {json.dumps({'sources': sources_data})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/")
async def index():
    """返回静态聊天页面"""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
