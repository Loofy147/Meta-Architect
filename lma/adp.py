import numpy as np
from scipy.stats import linregress
from typing import List, Dict
from lma.telemetry import TelemetrySnapshot


class ArchitecturalDynamicsPredictor:
    """Forecast future system degradation based on trends."""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def predict_entropy_trajectory(
        self, history: List[TelemetrySnapshot], coord: str, steps_ahead: int = 20
    ) -> Dict:
        """Predict future entropy values for a head."""
        if len(history) < 5:
            return {"error": "Insufficient history"}

        recent = history[-self.window_size :]
        entropies = [s.attention_entropy.get(coord, 0.9) for s in recent]
        steps = list(range(len(entropies)))

        if len(entropies) < 3:
            return {"error": "Insufficient data"}

        slope, intercept, r_value, p_value, std_err = linregress(steps, entropies)

        predictions = []
        for i in range(1, steps_ahead + 1):
            pred_step = len(entropies) + i
            pred_value = slope * pred_step + intercept
            predictions.append(
                {
                    "step": history[-1].step + i,
                    "predicted_entropy": max(0, min(1, pred_value)),
                    "confidence": abs(r_value),
                }
            )

        # Find when fixation threshold (0.3) would be crossed
        fixation_eta = None
        for pred in predictions:
            if pred["predicted_entropy"] < 0.3:
                fixation_eta = pred["step"]
                break

        return {
            "predictions": predictions,
            "trend_slope": slope,
            "confidence": abs(r_value),
            "fixation_eta": fixation_eta,
            "fixation_risk": "HIGH" if fixation_eta else "LOW",
        }

    def predict_t_mix_trajectory(
        self, history: List[TelemetrySnapshot], steps_ahead: int = 20
    ) -> Dict:
        """Predict future T_mix values."""
        if len(history) < 5:
            return {"error": "Insufficient history"}

        recent = history[-self.window_size :]
        t_mix_values = [s.t_mix for s in recent if s.t_mix > 0]
        steps = list(range(len(t_mix_values)))

        if len(t_mix_values) < 3:
            return {"error": "Insufficient data"}

        slope, intercept, r_value, p_value, std_err = linregress(steps, t_mix_values)

        predictions = []
        for i in range(1, steps_ahead + 1):
            pred_step = len(t_mix_values) + i
            pred_value = slope * pred_step + intercept
            predictions.append(
                {
                    "step": history[-1].step + i,
                    "predicted_t_mix": max(0, pred_value),
                    "confidence": abs(r_value),
                }
            )

        return {
            "predictions": predictions,
            "trend_slope": slope,
            "confidence": abs(r_value),
            "bottleneck_risk": (
                "HIGH" if slope > 0.5 else "MODERATE" if slope > 0 else "LOW"
            ),
        }
