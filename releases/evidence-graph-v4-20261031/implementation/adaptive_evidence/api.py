"""Synchronous AML contract with private validation errors and fail-closed auth."""
from __future__ import annotations

import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .core import Config, ConflictError, MemoryStore
from .remote_encoder import EmbeddingError, encoder_from_env
from . import __version__


class Message(BaseModel):
    role: str = Field(min_length=1)
    content: str = Field(min_length=1)
    timestamp: int | None = Field(default=None, strict=True)


class Add(BaseModel):
    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)


class Search(BaseModel):
    # Forbid accidental label/evidence fields in the public search contract.
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    options: list[str] | None = None
    top_k: int = Field(ge=1, le=100, strict=True)


@lru_cache(maxsize=1)
def get_store():
    encoder = encoder_from_env()
    try:
        return MemoryStore(os.getenv("AE_DATABASE", "data/context-weave-v310/memory.sqlite3"),
                           encoder, Config(context_chars=int(os.getenv("AE_CONTEXT_CHARS", "24000"))))
    except Exception:
        if hasattr(encoder, "close"):
            encoder.close()
        raise


def authenticate(authorization: str | None = Header(default=None), x_api_key: str | None = Header(default=None)):
    expected = os.getenv("MEMORY_API_KEY", "")
    if not expected:
        if os.getenv("AE_LOCAL_NO_AUTH") == "1":
            return
        raise HTTPException(503, "Memory service key is not configured")
    provided = x_api_key
    if not provided and authorization:
        scheme, _, credential = authorization.partition(" ")
        if scheme.lower() in {"bearer", "token"}:
            provided = credential
    if not provided or not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(401, "Invalid service key")


@asynccontextmanager
async def lifespan(app):
    if not os.getenv("MEMORY_API_KEY") and os.getenv("AE_LOCAL_NO_AUTH") != "1":
        raise RuntimeError("MEMORY_API_KEY is required before starting the service")
    store = get_store()
    try:
        yield
    finally:
        if hasattr(store.encoder, "close"):
            store.encoder.close()
        get_store.cache_clear()


LOG = logging.getLogger("adaptive_evidence")
LOG.setLevel(logging.INFO)
LOG.propagate = False
if not LOG.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOG.addHandler(handler)

app = FastAPI(title="Context Weave", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def request_timing(request, call_next):
    started = time.monotonic()
    response = await call_next(request)
    if request.url.path in {"/add", "/search"}:
        LOG.info("api_request path=%s status=%s seconds=%.2f", request.url.path,
                 response.status_code, time.monotonic() - started)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return JSONResponse(status_code=422, content={"detail": "Invalid request schema"})


@app.exception_handler(EmbeddingError)
async def embedding_error(request, exc):
    LOG.warning("embedding_failure reason=%s", str(exc))
    return JSONResponse(status_code=503, content={"detail": str(exc)}, headers={"Retry-After": "5"})


@app.get("/health")
def health(store: MemoryStore = Depends(get_store)):
    return {"status": "ok", "version": __version__,
            "embedding": getattr(store.encoder, "model", "local-or-test"),
            "embedding_batch_size": getattr(store.encoder, "batch_size", 10),
            "add_strategy": "raw_source", "search_strategy": os.getenv("AE_MODE", "query_document_evidence"),
            "llm": "none", "add_llm": "none", "search_planner": "none",
            "generative_llm_calls_in_memory_service": 0,
            "answer_owner": "external_evaluator", "context_chars": store.config.context_chars,
            "document_prior_rankings": store.document_prior_rankings,
            "document_reference_rankings": store.document_reference_rankings,
            "document_direct_rankings": store.document_direct_rankings,
            "conversation_rankings": store.conversation_rankings}


@app.post("/add", dependencies=[Depends(authenticate)])
def add(payload: Add, store: MemoryStore = Depends(get_store)):
    try:
        store.add(**payload.model_dump())
    except ConflictError:
        raise HTTPException(409, "Request ID conflicts with an existing write")
    except (ValueError, OverflowError, OSError):
        raise HTTPException(422, "Invalid memory data")
    return {"success": True, "request_id": payload.request_id,
            "user_id": payload.user_id, "session_id": payload.session_id}


@app.post("/search", dependencies=[Depends(authenticate)])
def search(payload: Search, store: MemoryStore = Depends(get_store)):
    try:
        return {"data": store.search(**payload.model_dump(), mode=os.getenv("AE_MODE", "query_document_evidence"))}
    except ValueError:
        raise HTTPException(422, "Invalid search data")
