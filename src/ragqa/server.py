from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .schemas import AnswerResult

# さっき作った資産をインポート（これがやりたかった！）
from .service import answer_question

app = FastAPI(title="Spec RAG QA API", description="仕様書QAシステム")


# リクエストボディの定義
class ChatRequest(BaseModel):
    query: str


@app.post("/api/v1/chat", response_model=AnswerResult)
def chat_endpoint(req: ChatRequest):
    """
    質問を受け取り、RAGを実行して結果を返します。
    """
    try:
        # 同期endpointとして、検索・LLM処理をFastAPIのスレッドプールで実行する。
        result = answer_question(req.query)
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=503, detail="Index not found. Please run ingest first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}
