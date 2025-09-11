"""
NeuralPolicyAdapter
===================

Optional neural network policy adapter exposing `predict_with_confidence` for
integration into the strategy ensemble. Keeps dependencies optional.
"""

from __future__ import annotations

from typing import Optional, Dict
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class NeuralPolicyAdapter:
    def __init__(self, model: Optional[object] = None):
        self.model = model
        self.ready = TORCH_AVAILABLE and (model is not None)

    def predict_with_confidence(self, features: np.ndarray) -> Optional[Dict[str, float]]:
        """Return {'signal': +/-1, 'confidence': [0,1]} or None if unavailable."""
        try:
            if not self.ready:
                return None
            x = features.astype(np.float32)
            with torch.no_grad():
                logits = self.model(torch.from_numpy(x).unsqueeze(0))  # shape [1, C]
                prob = torch.sigmoid(logits).item()
            signal = 1 if prob >= 0.5 else -1
            confidence = float(abs(prob - 0.5) * 2.0)
            return {"signal": signal, "confidence": confidence, "raw_prediction": prob}
        except Exception:
            return None

