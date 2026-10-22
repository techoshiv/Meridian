import logging
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from scripts.answer_generation import (
    OllamaConnectionError,
    OllamaRequestError,
    OllamaTimeoutError,
    answer_question,
)
from scripts.retrieval import retrieve


logger = logging.getLogger(__name__)
app = FastAPI(title="Meridian Banking RAG API")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, Field(min_length=1)]

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must not be blank")
        return question


class Citation(BaseModel):
    source_pdf: str
    pages: list[int]
    chunk_id: str | None = None
    section_ids: list[str] | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    found: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict[str, Any]:
    try:
        retrieval_result = retrieve(request.question)
    except Exception as error:
        logger.exception("Retrieval failed for /ask")
        raise HTTPException(status_code=503, detail="Retrieval service is unavailable.") from error

    try:
        return answer_question(request.question, retrieval_result)
    except OllamaTimeoutError as error:
        logger.warning("Ollama timed out for /ask: %s", error)
        raise HTTPException(status_code=504, detail="Answer generation timed out.") from error
    except OllamaConnectionError as error:
        logger.warning("Ollama connection failed for /ask: %s", error)
        raise HTTPException(status_code=503, detail="Answer generation service is unavailable.") from error
    except OllamaRequestError as error:
        logger.warning("Ollama request failed for /ask: %s", error)
        raise HTTPException(status_code=502, detail="Answer generation failed.") from error