# Critical for fusing temporal and factual outputs
import dspy
from dspy.teleprompt import COPRO

class UnifiedRewriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.factual_rewriter = dspy.ChainOfThought("query, metadata -> rewritten_query")
        self.causal_optimizer = COPRO(
            metric=lambda x: x.salience_score > 0.8,
            max_labeled_demos=5
        )

    def forward(self, query: str, source: str, context: dict):
        """Unifies temporal and factual processing paths"""
        if source == "TEMPORAL":
            # Inject Graphiti's causal edges for precision
            return self.causal_optimizer(
                query,
                context.get("graphiti_edges", []),
                n=3  # Generate 3 causal interpretations
            )
        
        # Optimize factual queries for speed + relevance
        return self.factual_rewriter(
            query=query,
            metadata=context.get("lightrag_metadata", {})
        )