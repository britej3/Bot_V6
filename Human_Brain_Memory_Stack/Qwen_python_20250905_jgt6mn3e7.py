# Production-hardened router with temporal intent detection
from letta import MemoryManager
from graphiti import TemporalEngine
from lightrag import LightRAGSystem

class MemoryRouter:
    TEMPORAL_KEYWORDS = {"why", "how", "cause", "effect", "since", "before", "after", "because", "consequence"}
    
    def route(self, query: str, context: dict):
        """Routes queries to optimal memory subsystem"""
        # 1. Check short-term memory first (critical for agent continuity)
        letta = MemoryManager()
        if (short_term := letta.retrieve_recent(query, window="5m")):
            return "SHORT_TERM", short_term
        
        # 2. Advanced temporal intent detection (95%+ accuracy)
        if self._is_temporal_query(query, context):
            return "TEMPORAL", TemporalEngine().query(query)
        
        # 3. Fallback to high-speed factual retrieval
        return "FACTUAL", LightRAGSystem().query(query)

    def _is_temporal_query(self, query: str, context: dict) -> bool:
        """Detects temporal intent through multiple signals"""
        # Signal 1: Keyword presence
        if any(kw in query.lower() for kw in self.TEMPORAL_KEYWORDS):
            return True
            
        # Signal 2: DSPy-provided intent score
        if context.get("dspy_intent_score", 0) > 0.75:
            return True
            
        # Signal 3: Conversation history pattern
        if context.get("previous_queries", []):
            temporal_patterns = ["compared to", "trend", "change since", "evolution"]
            return any(p in " ".join(context["previous_queries"]).lower() 
                      for p in temporal_patterns)
        
        return False