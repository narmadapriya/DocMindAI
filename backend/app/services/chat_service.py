from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List
from uuid import uuid4


class ChatService:
    """
    Phase 11 conversation service.

    Conversation history is isolated by user_id.

    The service deliberately does not modify the Phase 9
    vector database.
    """

    def __init__(
        self,
        chat_agent,
        retrieval_graph,
    ):
        self.chat_agent = chat_agent
        self.retrieval_graph = retrieval_graph

        self._lock = Lock()

        self._conversations: Dict[
            str,
            Dict[str, Any]
        ] = {}

    # =========================================================
    # CHAT
    # =========================================================

    def chat(
        self,
        *,
        user_id: str,
        query: str,
        document_ids: List[str],
        conversation_id: str | None = None,
        history: List[Dict[str, str]] | None = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:

        conversation_id = (
            conversation_id
            or str(uuid4())
        )

        history = list(
            history
            or self.get_history(
                user_id=user_id,
                conversation_id=conversation_id,
            )
        )

        retrieval = self.retrieval_graph.invoke(
            query=query,
            user_id=user_id,
            document_ids=document_ids,
            top_k=top_k,
        )

        evidence = list(
            retrieval.get("evidence", [])
        )

        validation = {
            "valid_evidence": evidence,
            "is_valid": bool(evidence),
            "complete": bool(evidence),
        }

        result = self.chat_agent.answer(
            query=query,
            evidence=evidence,
            history=history,
            validation=validation,
        )

        answer = result["answer"]

        self._append_message(
            user_id=user_id,
            conversation_id=conversation_id,
            role="user",
            content=query,
        )

        self._append_message(
            user_id=user_id,
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
        )

        result["conversation_id"] = (
            conversation_id
        )

        return result

    # =========================================================
    # HISTORY
    # =========================================================

    def get_history(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> List[Dict[str, str]]:

        with self._lock:

            conversation = (
                self._conversations.get(
                    conversation_id
                )
            )

            if not conversation:
                return []

            if (
                conversation["user_id"]
                != user_id
            ):
                return []

            return list(
                conversation["messages"]
            )

    def _append_message(
        self,
        *,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:

        with self._lock:

            conversation = (
                self._conversations.setdefault(
                    conversation_id,
                    {
                        "user_id": user_id,
                        "messages": [],
                    },
                )
            )

            if (
                conversation["user_id"]
                != user_id
            ):
                raise PermissionError(
                    "Conversation does not belong "
                    "to the requested user."
                )

            conversation["messages"].append(
                {
                    "role": role,
                    "content": content,
                }
            )