import numpy as np
from scipy.stats import linregress
from typing import List, Dict
from lma.telemetry import TelemetrySnapshot


class ArchitecturalDynamicsPredictor:
    """Forecasts future system degradation based on historical trends.

    The Architectural Dynamics Predictor (ADP) uses simple linear regression
    on a sliding window of historical telemetry data to predict the future
    trajectories of key metrics. This allows the LMA to anticipate potential
    issues like attention fixation or computational bottlenecks before they
    become critical.

    Attributes:
        window_size (int): The number of recent telemetry snapshots to use for
            linear regression.
    """

    def __init__(self, window_size: int = 20):
        """Initializes the ArchitecturalDynamicsPredictor.

        Args:
            window_size (int, optional): The size of the historical window for
                trend analysis. Defaults to 20.
        """
        self.window_size = window_size

    def predict_entropy_trajectory(
        self, history: List[TelemetrySnapshot], coord: str, steps_ahead: int = 20
    ) -> Dict:
        """Predicts the future entropy trajectory for a specific attention head.

        This method performs a linear regression on the recent entropy history
        of a single head to forecast its values for a specified number of future
        steps. It also estimates the risk of attention fixation.

        Args:
            history (List[TelemetrySnapshot]): The historical telemetry data.
            coord (str): The coordinate string of the head to be analyzed
                (e.g., "H(0,0)").
            steps_ahead (int, optional): The number of future steps to predict.
                Defaults to 20.

        Returns:
            Dict: A dictionary containing the predictions, trend analysis, and
                risk assessment. Returns a dictionary with an "error" key if
                there is insufficient data.
        """
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
        """Predicts the future trajectory of the GNN mixing time (t_mix).

        This method performs a linear regression on the recent history of
        `t_mix` to forecast its future values. It also provides a qualitative
        assessment of the risk of the GNN mixer becoming a computational
        bottleneck.

        Args:
            history (List[TelemetrySnapshot]): The historical telemetry data.
            steps_ahead (int, optional): The number of future steps to predict.
                Defaults to 20.

        Returns:
            Dict: A dictionary containing the predictions, trend analysis, and
                risk assessment. Returns a dictionary with an "error" key if
                there is insufficient data.
        """
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
