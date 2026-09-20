from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypedDict,
)

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.rag.agents.router_agent import (
    RouterAgent,
)

from app.rag.agents.retrieval_agent import (
    RetrievalAgent,
)

from app.rag.agents.evidence_validation_agent import (
    EvidenceValidationAgent,
)

from app.rag.agents.reasoning_agent import (
    ReasoningAgent,
)

from app.rag.agents.verification_agent import (
    VerificationAgent,
)

from app.rag.agents.citation_agent import (
    CitationAgent,
)


# ============================================================
# PHASE 9 STATE
# ============================================================


class RetrievalState(
    TypedDict,
    total=False,
):

    query: str
    user_id: str

    document_ids: List[str]

    metadata_filters: Dict[str, Any]

    top_k: int

    plan: Dict[str, Any]

    rewritten_query: str

    intent: str

    scope: str

    modalities: List[str]

    evidence: List[
        Dict[str, Any]
    ]

    evidence_count: int

    retrieval_status: str


# ============================================================
# PHASE 9 RETRIEVAL GRAPH
# ============================================================


class RetrievalGraph:
    """
    Phase 9 Retrieval Graph.

    This workflow is intentionally preserved.

        Router
          ↓
        Planning
          ↓
        Rewrite
          ↓
        Retrieval
    """

    def __init__(
        self,
        router_agent: Optional[
            RouterAgent
        ] = None,
        retrieval_agent: Optional[
            RetrievalAgent
        ] = None,
    ):

        self.router = (
            router_agent
            if router_agent is not None
            else RouterAgent()
        )

        self.retrieval = (
            retrieval_agent
            if retrieval_agent is not None
            else RetrievalAgent()
        )

        self.graph = (
            self._build_graph()
        )

    def _build_graph(self):

        builder = StateGraph(
            RetrievalState
        )

        builder.add_node(
            "router",
            self.router_node,
        )

        builder.add_node(
            "query_planning",
            self.query_planning_node,
        )

        builder.add_node(
            "query_rewrite",
            self.query_rewrite_node,
        )

        builder.add_node(
            "retriever",
            self.retriever_node,
        )

        builder.add_edge(
            START,
            "router",
        )

        builder.add_edge(
            "router",
            "query_planning",
        )

        builder.add_edge(
            "query_planning",
            "query_rewrite",
        )

        builder.add_edge(
            "query_rewrite",
            "retriever",
        )

        builder.add_edge(
            "retriever",
            END,
        )

        return builder.compile()

    def router_node(
        self,
        state: RetrievalState,
    ) -> Dict[str, Any]:

        plan = self.router.plan(
            state["query"],
            user_id=state[
                "user_id"
            ],
            document_ids=state.get(
                "document_ids"
            ),
            top_k=state.get(
                "top_k"
            ),
            metadata_filters=state.get(
                "metadata_filters"
            ),
        )

        return {
            "plan": {
                "original_query":
                    plan.original_query,

                "rewritten_query":
                    plan.rewritten_query,

                "intent":
                    plan.intent,

                "scope":
                    plan.scope,

                "document_ids":
                    plan.document_ids,

                "modalities":
                    plan.modalities,

                "metadata_filters":
                    plan.metadata_filters,

                "top_k":
                    plan.top_k,

                "min_relevance":
                    plan.min_relevance,
            }
        }

    def query_planning_node(
        self,
        state: RetrievalState,
    ) -> Dict[str, Any]:

        plan = state[
            "plan"
        ]

        return {
            "intent":
                plan["intent"],

            "scope":
                plan["scope"],

            "modalities":
                plan["modalities"],
        }

    def query_rewrite_node(
        self,
        state: RetrievalState,
    ) -> Dict[str, Any]:

        plan = state[
            "plan"
        ]

        return {
            "rewritten_query":
                plan.get(
                    "rewritten_query",
                    state["query"],
                )
        }

    def retriever_node(
        self,
        state: RetrievalState,
    ) -> Dict[str, Any]:

        plan = state[
            "plan"
        ]

        evidence = (
            self.retrieval.retrieve(
                query=state[
                    "rewritten_query"
                ],

                user_id=state[
                    "user_id"
                ],

                document_ids=plan.get(
                    "document_ids",
                    [],
                ),

                modalities=plan.get(
                    "modalities",
                    [],
                ),

                metadata_filters=plan.get(
                    "metadata_filters",
                    {},
                ),

                top_k=plan.get(
                    "top_k",
                    state.get(
                        "top_k",
                        5,
                    ),
                ),

                min_relevance=plan.get(
                    "min_relevance",
                    0.20,
                ),
            )
        )

        return {
            "evidence":
                evidence,

            "evidence_count":
                len(evidence),

            "retrieval_status":
                (
                    "success"
                    if evidence
                    else "no_evidence"
                ),
        }

    def invoke(
        self,
        *,
        query: str,
        user_id: str,
        document_ids: Optional[
            List[str]
        ] = None,
        metadata_filters: Optional[
            Dict[str, Any]
        ] = None,
        top_k: int = 5,
    ) -> RetrievalState:

        if (
            not query
            or not query.strip()
        ):
            raise ValueError(
                "Query cannot be empty."
            )

        if (
            not user_id
            or not user_id.strip()
        ):
            raise ValueError(
                "user_id cannot be empty."
            )

        state: RetrievalState = {
            "query":
                query.strip(),

            "user_id":
                user_id,

            "document_ids":
                document_ids or [],

            "metadata_filters":
                metadata_filters or {},

            "top_k":
                top_k,
        }

        return self.graph.invoke(
            state
        )


# ============================================================
# PHASE 10 STATE
# ============================================================


class Phase10State(
    TypedDict,
    total=False,
):

    query: str
    user_id: str

    document_ids: List[str]

    metadata_filters: Dict[str, Any]

    top_k: int

    rewritten_query: str

    intent: str

    scope: str

    modalities: List[str]

    evidence: List[
        Dict[str, Any]
    ]

    evidence_count: int

    retrieval_status: str

    validation: Dict[str, Any]

    validation_status: str

    reasoning: Dict[str, Any]

    reasoning_status: str

    verification: Dict[str, Any]

    verification_status: str

    citations: List[
        Dict[str, Any]
    ]

    citation_count: int

    answer: str

    final_status: str

    retry_count: int

    max_retries: int


# ============================================================
# PHASE 10 GRAPH
# ============================================================


class MultimodalReasoningGraph:
    """
    Phase 10 Multimodal Reasoning Graph.

    START
       ↓
    Phase 9 Retrieval
       ↓
    Evidence Validation
       ↓
    Qwen2.5-VL Reasoning
       ↓
    Verification
       ↓
    ┌──────────────┐
    │              │
    PASS           FAIL
    │              │
    ↓              ↓
    Citation     Query Rewrite
    │              │
    ↓              ↓
    Final        Retrieval
    """

    def __init__(
        self,
        retrieval_graph:
            Optional[
                RetrievalGraph
            ] = None,

        validation_agent:
            Optional[
                EvidenceValidationAgent
            ] = None,

        reasoning_agent:
            Optional[
                ReasoningAgent
            ] = None,

        verification_agent:
            Optional[
                VerificationAgent
            ] = None,

        citation_agent:
            Optional[
                CitationAgent
            ] = None,

        max_retries: int = 1,
    ):

        self.retrieval_graph = (
            retrieval_graph
            or RetrievalGraph()
        )

        self.validator = (
            validation_agent
            or EvidenceValidationAgent()
        )

        self.reasoner = (
            reasoning_agent
            or ReasoningAgent()
        )

        self.verifier = (
            verification_agent
            or VerificationAgent()
        )

        self.citation = (
            citation_agent
            or CitationAgent()
        )

        self.max_retries = max(
            0,
            int(max_retries),
        )

        self.graph = (
            self._build_graph()
        )

    # ========================================================
    # BUILD
    # ========================================================

    def _build_graph(self):

        builder = StateGraph(
            Phase10State
        )

        builder.add_node(
            "retrieval",
            self.retrieval_node,
        )

        builder.add_node(
            "evidence_validation",
            self.validation_node,
        )

        builder.add_node(
            "reasoning",
            self.reasoning_node,
        )

        builder.add_node(
            "verification",
            self.verification_node,
        )

        builder.add_node(
            "query_rewrite_retry",
            self.query_rewrite_retry_node,
        )

        builder.add_node(
            "citation",
            self.citation_node,
        )

        builder.add_node(
            "final_failure",
            self.final_failure_node,
        )

        builder.add_edge(
            START,
            "retrieval",
        )

        builder.add_edge(
            "retrieval",
            "evidence_validation",
        )

        builder.add_edge(
            "evidence_validation",
            "reasoning",
        )

        builder.add_edge(
            "reasoning",
            "verification",
        )

        builder.add_conditional_edges(
            "verification",
            self.route_after_verification,
            {
                "citation":
                    "citation",

                "retry":
                    "query_rewrite_retry",

                "failure":
                    "final_failure",
            },
        )

        builder.add_edge(
            "query_rewrite_retry",
            "retrieval",
        )

        builder.add_edge(
            "citation",
            END,
        )

        builder.add_edge(
            "final_failure",
            END,
        )

        return builder.compile()

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieval_node(
        self,
        state: Phase10State,
    ) -> Dict[str, Any]:

        result = (
            self.retrieval_graph.invoke(
                query=(
                    state.get(
                        "rewritten_query"
                    )
                    or state["query"]
                ),

                user_id=state[
                    "user_id"
                ],

                document_ids=state.get(
                    "document_ids",
                    [],
                ),

                metadata_filters=state.get(
                    "metadata_filters",
                    {},
                ),

                top_k=state.get(
                    "top_k",
                    5,
                ),
            )
        )

        return {
            "rewritten_query":
                result.get(
                    "rewritten_query",
                    result["query"],
                ),

            "intent":
                result.get(
                    "intent"
                ),

            "scope":
                result.get(
                    "scope"
                ),

            "modalities":
                result.get(
                    "modalities",
                    [],
                ),

            "evidence":
                result.get(
                    "evidence",
                    [],
                ),

            "evidence_count":
                result.get(
                    "evidence_count",
                    0,
                ),

            "retrieval_status":
                result.get(
                    "retrieval_status",
                    "unknown",
                ),
        }

    # ========================================================
    # VALIDATION
    # ========================================================

    def validation_node(
        self,
        state: Phase10State,
    ) -> Dict[str, Any]:

        validation = (
            self.validator.validate(
                state[
                    "rewritten_query"
                ],
                state.get(
                    "evidence",
                    [],
                ),
            )
        )

        return {
            "validation":
                validation,

            "validation_status":
                validation[
                    "status"
                ],
        }

    # ========================================================
    # REASONING
    # ========================================================

    def reasoning_node(
        self,
        state: Phase10State,
    ) -> Dict[str, Any]:

        reasoning = (
            self.reasoner.reason(
                state[
                    "rewritten_query"
                ],

                state.get(
                    "evidence",
                    [],
                ),

                validation=state.get(
                    "validation"
                ),
            )
        )

        return {
            "reasoning":
                reasoning,

            "reasoning_status":
                reasoning.get(
                    "status",
                    "unknown",
                ),
        }

    # ========================================================
    # VERIFICATION
    # ========================================================

    def verification_node(
        self,
        state: Phase10State,
    ) -> Dict[str, Any]:

        verification = (
            self.verifier.verify(
                state[
                    "rewritten_query"
                ],

                state.get(
                    "reasoning",
                    {},
                ),

                state.get(
                    "evidence",
                    [],
                ),

                validation=state.get(
                    "validation"
                ),
            )
        )

        return {
            "verification":
                verification,

            "verification_status":
                verification[
                    "status"
                ],
        }

    # ========================================================
    # ROUTING
    # ========================================================

    def route_after_verification(
        self,
        state: Phase10State,
    ) -> str:

        verification = (
            state.get(
                "verification",
                {},
            )
        )

        if (
            verification.get(
                "passed"
            )
            is True
        ):

            return "citation"

        if (
            int(
                state.get(
                    "retry_count",
                    0,
                )
            )
            <
            int(
                state.get(
                    "max_retries",
                    self.max_retries,
                )
            )
        ):

            return "retry"

        return "failure"

    # ========================================================
    # QUERY REWRITE
    # ========================================================

    def query_rewrite_retry_node(
        self,
        state: Phase10State,
    ) -> Dict[str, Any]:

        verification = (
            state.get(
                "verification",
                {},
            )
        )

        rewritten = (
            verification.get(
                "rewritten_query"
            )
            or (
                f"{state['query']} "
                "Use only directly "
                "supported facts and "
                "cite the evidence."
            )
        )

        return {
            "rewritten_query":
                rewritten,

            "retry_count":
                int(
                    state.get(
                        "retry_count",
                        0,
                    )
                )
                + 1,
        }

    # ========================================================
    # CITATION
    # ========================================================

    def citation_node(
        self,
        state: Phase10State,
    ) -> Dict[str, Any]:

        reasoning = (
            state.get(
                "reasoning",
                {},
            )
        )

        cited = (
            self.citation.cite(
                reasoning.get(
                    "answer",
                    "",
                ),

                state.get(
                    "evidence",
                    [],
                ),

                reasoning.get(
                    "used_evidence_ids",
                    [],
                ),
            )
        )

        return {
            "answer":
                cited["answer"],

            "citations":
                cited["citations"],

            "citation_count":
                cited[
                    "citation_count"
                ],

            "final_status":
                "success",
        }

    # ========================================================
    # FINAL FAILURE
    # ========================================================

    def final_failure_node(
        self,
        state: Phase10State,
    ) -> Dict[str, Any]:

        verification = (
            state.get(
                "verification",
                {},
            )
        )

        return {
            "answer": (
                "I could not produce "
                "a sufficiently "
                "verified answer from "
                "the retrieved evidence."
            ),

            "citations": [],

            "citation_count": 0,

            "final_status":
                "verification_failed",

            "verification_status":
                verification.get(
                    "status",
                    "fail",
                ),
        }

    # ========================================================
    # PUBLIC INVOKE
    # ========================================================

    def invoke(
        self,
        *,
        query: str,
        user_id: str,
        document_ids: Optional[
            List[str]
        ] = None,
        metadata_filters: Optional[
            Dict[str, Any]
        ] = None,
        top_k: int = 5,
        max_retries: Optional[
            int
        ] = None,
    ) -> Phase10State:

        if (
            not query
            or not query.strip()
        ):
            raise ValueError(
                "Query cannot be empty."
            )

        if (
            not user_id
            or not user_id.strip()
        ):
            raise ValueError(
                "user_id cannot be empty."
            )

        retries = (
            self.max_retries
            if max_retries is None
            else max(
                0,
                int(
                    max_retries
                ),
            )
        )

        state: Phase10State = {
            "query":
                query.strip(),

            "user_id":
                user_id,

            "document_ids":
                document_ids or [],

            "metadata_filters":
                metadata_filters or {},

            "top_k":
                top_k,

            "retry_count":
                0,

            "max_retries":
                retries,
        }

        return self.graph.invoke(
            state
        )


# ============================================================
# FACTORIES
# ============================================================


def build_retrieval_graph(
    router_agent: Optional[
        RouterAgent
    ] = None,
    retrieval_agent: Optional[
        RetrievalAgent
    ] = None,
) -> RetrievalGraph:

    return RetrievalGraph(
        router_agent=router_agent,
        retrieval_agent=retrieval_agent,
    )


def build_multimodal_reasoning_graph(
    **kwargs: Any,
) -> MultimodalReasoningGraph:

    return MultimodalReasoningGraph(
        **kwargs
    )