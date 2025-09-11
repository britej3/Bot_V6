"""
Embedding DAO (baseline, portable)
==================================

Stores and retrieves vector embeddings and metadata using SQLAlchemy models.
Portable across SQLite/Postgres; can later migrate to pgvector extension.
Includes a simple in-Python cosine similarity for top-k queries (baseline).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

from sqlalchemy import select, insert, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.database.manager import DatabaseManager
from src.database.models import EmbeddingRecord


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (na * nb)


@dataclass
class EmbeddingItem:
    id: int
    namespace: str
    item_id: str
    vector: List[float]
    metadata: Dict[str, Any]


class EmbeddingDAO:
    """Data access for embeddings and similarity search (baseline)."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    async def upsert_embedding(self, namespace: str, item_id: str,
                               vector: List[float],
                               metadata: Optional[Dict[str, Any]] = None) -> int:
        """Create or update an embedding record; returns record id."""
        metadata = metadata or {}

        # Use dialect-specific upsert where possible; fallback to manual
        stmt = None
        if self.db.db_url.startswith("postgresql") or self.db.db_url.startswith("postgres"):
            stmt = pg_insert(EmbeddingRecord.__table__).values(
                namespace=namespace, item_id=item_id, vector=vector, metadata_=metadata
            ).on_conflict_do_update(
                index_elements=[EmbeddingRecord.namespace, EmbeddingRecord.item_id],
                set_={"vector": vector, "metadata": metadata}
            ).returning(EmbeddingRecord.id)
        elif self.db.db_url.startswith("sqlite"):
            # SQLite 3.24+ supports upsert
            stmt = sqlite_insert(EmbeddingRecord.__table__).values(
                namespace=namespace, item_id=item_id, vector=vector, metadata_=metadata
            ).on_conflict_do_update(
                index_elements=[EmbeddingRecord.namespace, EmbeddingRecord.item_id],
                set_={"vector": vector, "metadata": metadata}
            )
        else:
            # Generic: try update then insert
            async with self.db.get_session() as session:
                res = await session.execute(
                    update(EmbeddingRecord)
                    .where(EmbeddingRecord.namespace == namespace,
                           EmbeddingRecord.item_id == item_id)
                    .values(vector=vector, metadata_=metadata)
                    .returning(EmbeddingRecord.id)
                )
                row = res.first()
                if row:
                    return row[0]
                res = await session.execute(
                    insert(EmbeddingRecord).values(
                        namespace=namespace, item_id=item_id, vector=vector, metadata_=metadata
                    ).returning(EmbeddingRecord.id)
                )
                return res.first()[0]

        async with self.db.get_session() as session:
            res = await session.execute(stmt)
            row = res.first()
            if row is None:
                # For SQLite, returning may not work; fetch id
                q = await session.execute(
                    select(EmbeddingRecord.id)
                    .where(EmbeddingRecord.namespace == namespace,
                           EmbeddingRecord.item_id == item_id)
                )
                row = q.first()
            return int(row[0])

    async def get_embedding(self, namespace: str, item_id: str) -> Optional[EmbeddingItem]:
        async with self.db.get_session() as session:
            res = await session.execute(
                select(EmbeddingRecord)
                .where(EmbeddingRecord.namespace == namespace,
                       EmbeddingRecord.item_id == item_id)
            )
            row = res.scalar_one_or_none()
            if not row:
                return None
            return EmbeddingItem(
                id=row.id,
                namespace=row.namespace,
                item_id=row.item_id,
                vector=list(row.vector or []),
                metadata=dict(row.metadata_ or {}),
            )

    async def query_top_k(self, namespace: str, query_vector: List[float], k: int = 5) -> List[Tuple[EmbeddingItem, float]]:
        """Baseline in-Python cosine similarity over all embeddings in a namespace.

        Replace with pgvector ANN when available.
        """
        async with self.db.get_session() as session:
            res = await session.execute(
                select(EmbeddingRecord)
                .where(EmbeddingRecord.namespace == namespace)
            )
            rows = res.scalars().all()

        scored: List[Tuple[EmbeddingItem, float]] = []
        for r in rows:
            item = EmbeddingItem(
                id=r.id,
                namespace=r.namespace,
                item_id=r.item_id,
                vector=list(r.vector or []),
                metadata=dict(r.metadata_ or {}),
            )
            score = _cosine(item.vector, query_vector)
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

