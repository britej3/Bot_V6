"""
MemoryService Router
====================

Routes memory operations to appropriate stores:
- Short-term working memory (Redis) with TTL and bounded lists
- Vector embeddings via EmbeddingDAO (portable; future pgvector/ANN)
- Graph queries (placeholder; gated via feature flag)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable
import json

from src.database.redis_manager import RedisManager
from src.database.manager import DatabaseManager
from src.database.embedding_dao import EmbeddingDAO, EmbeddingItem


class MemoryService:
    def __init__(self,
                 redis_manager: RedisManager,
                 db_manager: DatabaseManager,
                 default_ttl_seconds: int = 300,
                 max_list_length: int = 100,
                 enable_graph: bool = False):
        self.redis = redis_manager
        self.db = db_manager
        self.dao = EmbeddingDAO(db_manager)
        self.default_ttl = max(1, int(default_ttl_seconds))
        self.max_list_length = max(1, int(max_list_length))
        self.enable_graph = enable_graph

    # --- Short-term working memory (Redis) ---
    async def put_short_term(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        ttl = int(ttl or self.default_ttl)
        payload = json.dumps(value)
        return await self.redis.set_json(f"wm:{key}", payload, ttl_seconds=ttl)

    async def get_short_term(self, key: str) -> Optional[Any]:
        raw = await self.redis.get_json(f"wm:{key}")
        return json.loads(raw) if raw else None

    async def push_recent(self, list_name: str, item: Any) -> int:
        """Push to a bounded recent list (LPUSH + LTRIM)."""
        payload = json.dumps(item)
        return await self.redis.lpush_bounded(f"wm:list:{list_name}", payload, self.max_list_length, ttl_seconds=self.default_ttl)

    async def get_recent(self, list_name: str, count: int = 10) -> List[Any]:
        items = await self.redis.lrange_json(f"wm:list:{list_name}", 0, count - 1)
        out: List[Any] = []
        for raw in items:
            try:
                out.append(json.loads(raw))
            except Exception:
                continue
        return out

    # --- Embeddings (vector) ---
    async def upsert_embedding(self, namespace: str, item_id: str,
                               vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> int:
        return await self.dao.upsert_embedding(namespace, item_id, vector, metadata)

    async def query_similar(self, namespace: str, vector: List[float], k: int = 5) -> List[Tuple[EmbeddingItem, float]]:
        return await self.dao.query_top_k(namespace, vector, k)

    # --- Graph (placeholder) ---
    async def graph_query(self, query: str) -> Dict[str, Any]:
        if not self.enable_graph:
            return {"error": "graph disabled"}
        # Implement Graphiti/Neo4j calls in a separate module/service
        return {"error": "graph not implemented"}

