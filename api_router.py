"""
REST API 模块（使用FastAPI实现）
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import re  # 添加正则模块导入
from typing import Dict, Any
import logging
from combined_rag import process_pdf, combined_query_answer

app = FastAPI(
    title="本地RAG API服务",
    description="提供基于本地大模型的文档问答API接口",
    version="1.0.0"
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    enable_web: bool = True

class AnswerResponse(BaseModel):
    answer: str
    sources: list[Dict[str, Any]]
    confidence: float

@app.post("/api/upload", summary="上传PDF文档")
async def upload_pdf(file: UploadFile = File(...)):
    """
    处理PDF文档并存入向量数据库
    - 支持格式：application/pdf
    - 最大文件大小：50MB
    """
    if file.content_type != "application/pdf":
        raise HTTPException(400, "仅支持PDF文件")

    try:
        # 保存临时文件
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 调用现有处理逻辑
        result = process_pdf(tmp_path, _dummy_progress)
        os.unlink(tmp_path)  # 清理临时文件

        return {"status": "success", "chunks": result.split(" ")[3]}
    except Exception as e:
        logging.error(f"PDF处理失败: {str(e)}")
        raise HTTPException(500, "文档处理失败") from e

@app.post("/api/ask", response_model=AnswerResponse, summary="提出问题")
async def ask_question(req: QuestionRequest):
    """
    问答接口
    - question: 问题内容
    - enable_web: 是否启用网络搜索增强（默认True）
    """
    try:
        # 调用现有问答逻辑
        answer_html = combined_query_answer(req.question, _dummy_progress)
        
        # 从HTML提取纯文本和来源（需适配现有格式）
        answer_text = re.sub('<[^<]+?>', '', answer_html)
        sources = re.findall(r'data-source="([^"]+)"', answer_html)
        
        return {
            "answer": answer_text,
            "sources": [{"source": s} for s in sources],
            "confidence": 0.8  # 示例置信度，实际可添加评估逻辑
        }
    except Exception as e:
        logging.error(f"问答失败: {str(e)}")
        raise HTTPException(500, "问答处理失败") from e

def _dummy_progress(percent: float, desc: str = None):
    """替代Gradio进度回调的虚拟方法"""
    logging.info(f"[进度 {percent*100:.1f}%] {desc}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=17995) 