# Context fusion engine (patch Cogniee v0.8+)
class ContextAssembler:
    def __init__(self):
        # Critical: Enable temporal signal processing
        self.cogniee = Cogniee(enable_temporal_fusion=True)
    
    def assemble(self, dspy_output: dict, source: str):
        """Creates human-like contextual understanding"""
        # Normalize all inputs to unified schema
        normalized = {
            "temporal_signals": dspy_output.get("causal_chains", []) if source == "TEMPORAL" else [],
            "factual_data": dspy_output["results"],
            "salience_score": dspy_output.get("salience", 0.5)
        }
        
        # Apply brain-inspired fusion
        return self.cogniee.fuse(
            primary=normalized["factual_data"],
            temporal_context=normalized["temporal_signals"],
            salience=normalized["salience_score"],
            fusion_strategy="hippocampal"  # Activates memory integration
        )