from __future__ import annotations

from typing import Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.analytics import Analytics


class AnalyticsService:
    """
    PostgreSQL analytics persistence for DocMindAI RAG actions.

    Step 9 records:

        rag_chat
        rag_summary
        rag_comparison

    One Analytics row represents one completed user operation.

    document_id is populated only when exactly one document
    was selected. Multi-document operations intentionally use
    NULL because the frozen analytics schema has only one
    document_id column.
    """

    ALLOWED_EVENTS = {
        "rag_chat",
        "rag_summary",
        "rag_comparison",
    }

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    # =========================================================
    # RECORD EVENT
    # =========================================================

    def record_event(
        self,
        *,
        user_id: UUID | str,
        event_type: str,
        document_ids: Sequence[
            UUID | str
        ] | None = None,
        query_count: int = 1,
        response_time_ms: int | None = None,
        commit: bool = True,
    ) -> Analytics:

        if event_type not in self.ALLOWED_EVENTS:
            raise ValueError(
                f"Unsupported analytics event: "
                f"{event_type}"
            )

        user_uuid = self._parse_uuid(
            user_id,
            field_name="user_id",
        )

        normalized_documents = [
            self._parse_uuid(
                value,
                field_name="document_id",
            )
            for value
            in (
                document_ids
                or []
            )
        ]

        # Frozen Analytics model can reference only one
        # document. Multi-document actions therefore use NULL.
        document_id = (
            normalized_documents[0]
            if len(
                normalized_documents
            ) == 1
            else None
        )

        event = Analytics(
            user_id=user_uuid,
            event_type=event_type,
            document_id=document_id,
            query_count=max(
                0,
                int(query_count),
            ),
            response_time_ms=(
                None
                if response_time_ms
                is None
                else max(
                    0,
                    int(
                        response_time_ms
                    ),
                )
            ),
        )

        try:

            self.db.add(
                event
            )

            self.db.flush()

            if commit:

                self.db.commit()

                self.db.refresh(
                    event
                )

            return event

        except Exception:

            if commit:
                self.db.rollback()

            raise

    # =========================================================
    # READ
    # =========================================================

    def get_user_events(
        self,
        user_id: UUID | str,
        *,
        event_type: str | None = None,
    ) -> list[Analytics]:

        user_uuid = self._parse_uuid(
            user_id,
            field_name="user_id",
        )

        query = (
            self.db.query(
                Analytics
            )
            .filter(
                Analytics.user_id
                == user_uuid
            )
        )

        if event_type:

            query = query.filter(
                Analytics.event_type
                == event_type
            )

        return (
            query
            .order_by(
                Analytics.created_at.asc()
            )
            .all()
        )

    def count_user_events(
        self,
        user_id: UUID | str,
        *,
        event_type: str | None = None,
    ) -> int:

        user_uuid = self._parse_uuid(
            user_id,
            field_name="user_id",
        )

        query = (
            self.db.query(
                Analytics
            )
            .filter(
                Analytics.user_id
                == user_uuid
            )
        )

        if event_type:

            query = query.filter(
                Analytics.event_type
                == event_type
            )

        return query.count()

    # =========================================================
    # UUID
    # =========================================================

    @staticmethod
    def _parse_uuid(
        value: UUID | str,
        *,
        field_name: str,
    ) -> UUID:

        if isinstance(
            value,
            UUID,
        ):
            return value

        try:

            return UUID(
                str(value)
            )

        except (
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:

            raise ValueError(
                f"{field_name} must be "
                "a valid UUID."
            ) from exc