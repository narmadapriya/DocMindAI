from __future__ import annotations

from typing import Any, Dict, Iterable, List


class CitationWhitelist:
    """
    Citation whitelist.

    Only evidence returned by retrieval can become
    a citation.
    """

    def build(
        self,
        evidence: Iterable[
            Dict[str, Any]
        ],
    ) -> Dict[
        str,
        Dict[str, Any],
    ]:

        whitelist = {}

        for index, item in enumerate(
            evidence,
            1,
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            evidence_id = str(
                item.get(
                    "chunk_id"
                )
                or f"evidence-{index}"
            )

            whitelist[
                evidence_id
            ] = {
                "evidence_id":
                    evidence_id,
                "metadata":
                    item.get(
                        "metadata"
                    )
                    or {},
                "content":
                    item.get(
                        "content",
                        "",
                    ),
            }

        return whitelist

    def allowed(
        self,
        evidence_id: str,
        whitelist: Dict[
            str,
            Dict[str, Any],
        ],
    ) -> bool:

        return (
            evidence_id
            in whitelist
        )

    def select(
        self,
        evidence_ids: Iterable[
            str
        ],
        whitelist: Dict[
            str,
            Dict[str, Any],
        ],
    ) -> List[
        Dict[str, Any]
    ]:

        return [
            whitelist[
                evidence_id
            ]
            for evidence_id
            in evidence_ids
            if evidence_id
            in whitelist
        ]