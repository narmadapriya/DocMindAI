from __future__ import annotations

from functools import lru_cache
import os
import threading

from app.rag.agents.chat_agent import (
    ChatAgent,
)
from app.rag.agents.summary_agent import (
    SummaryAgent,
)
from app.rag.agents.comparison_agent import (
    ComparisonAgent,
)

from app.rag.agents.reasoning_agent import (
    ReasoningAgent,
)

from app.rag.agents.graph import (
    RetrievalGraph,
)

from app.rag.pipelines.retrieval_pipeline import (
    RetrievalPipeline,
)

from app.rag.llm.ollama_client import (
    OllamaClient,
)

from app.core.logging import (
    get_logger,
    log_event,
)

from app.services.chat_service import (
    ChatService,
)

from app.services.summary_service import (
    SummaryService,
)

from app.services.comparison_service import (
    ComparisonService,
)


logger = get_logger(__name__)


def _env_bool(
    name: str,
    default: bool = True,
) -> bool:
    value = os.getenv(
        name,
        "true" if default else "false",
    )
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@lru_cache(maxsize=1)
def get_llm_client():
    """
    Return the single process-wide Qwen client.

    Reusing this client preserves the Ollama HTTP/model keep-alive path
    across requests and avoids recreating the local inference client on
    every Chat & Ask turn.
    """

    return OllamaClient(
        model="qwen2.5vl:3b",
    )


@lru_cache(maxsize=1)
def get_reasoning_agent():
    """Return the single process-wide reasoning agent."""

    return ReasoningAgent(
        llm_client=get_llm_client(),
        model="qwen2.5vl:3b",
    )


@lru_cache(maxsize=1)
def get_retrieval_graph():
    """
    Return the single process-wide Phase 9 retrieval graph.

    The graph itself contains no SQLAlchemy Session. Reusing it is safe
    for the existing request-scoped PostgreSQL verification boundary and
    allows its EmbeddingModel, Ollama embedding client, Chroma client and
    Chroma collection handles to stay warm between requests.
    """

    return RetrievalGraph()


@lru_cache(maxsize=1)
def get_retrieval_pipeline():
    """
    Reuse the process-wide retrieval infrastructure while keeping the
    application-level RetrievalService request/DB scoped.
    """

    return RetrievalPipeline(
        retrieval_graph=get_retrieval_graph(),
    )


def _warm_reasoning_model() -> None:
    """
    Preserve the frozen Phase-15 startup behaviour: warm only the
    Qwen reasoning model. Retrieval/embedding warm-up is deliberately
    not duplicated here because background ingestion already owns its
    embedding warm-up path.

    This keeps Chat & Ask warm-start behaviour without competing with
    document ingestion for the local Ollama embedding model/Chroma
    resources on an 8-GB machine.
    """

    if not _env_bool(
        "RAG_REASONING_WARMUP_ENABLED",
        True,
    ):
        return

    try:
        client = get_llm_client()

        if (
            not client.is_available()
            or not client.is_model_available()
        ):
            log_event(
                logger,
                "reasoning_warmup_skipped",
                model=client.model,
                reason="ollama_or_model_unavailable",
            )
            return

        client.chat(
            prompt=(
                'Return only this JSON object: '
                '{"status":"ready"}'
            ),
            images=[],
            temperature=0.0,
            num_predict=16,
            num_ctx=1024,
            json_mode=True,
            keep_alive=os.getenv(
                "OLLAMA_KEEP_ALIVE",
                "30m",
            ),
        )

        log_event(
            logger,
            "reasoning_warmup",
            model=client.model,
            status="ready",
        )

    except Exception as exc:
        # Warm-up must never prevent FastAPI from starting. The existing
        # Chat path remains capable of loading Ollama normally on demand.
        log_event(
            logger,
            "reasoning_warmup_failed",
            error=str(exc),
        )


def _start_reasoning_warmup() -> None:
    thread = threading.Thread(
        target=_warm_reasoning_model,
        name="docmindai-reasoning-warmup",
        daemon=True,
    )
    thread.start()


_start_reasoning_warmup()


@lru_cache(maxsize=1)
def get_chat_service():

    agent = ChatAgent(
        reasoning_agent=get_reasoning_agent(),
    )

    return ChatService(
        chat_agent=agent,
        retrieval_graph=get_retrieval_graph(),
    )


@lru_cache(maxsize=1)
def get_summary_service():

    agent = SummaryAgent(
        reasoning_agent=get_reasoning_agent(),
    )

    return SummaryService(
        summary_agent=agent,
        retrieval_graph=get_retrieval_graph(),
    )


@lru_cache(maxsize=1)
def get_comparison_service():

    agent = ComparisonAgent(
        reasoning_agent=get_reasoning_agent(),
    )

    return ComparisonService(
        comparison_agent=agent,
        retrieval_graph=get_retrieval_graph(),
    )