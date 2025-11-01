# Standard Library imports
from dataclasses import dataclass
from math import hypot
from typing import Optional, TypedDict, Protocol

# External imports
import numpy as np

# TODO
# There seem to be two kind of anomalies
# - threshold spikes
# - ZScore spikes
# Allow the property of interest to be a parameter.
# This requires all the properties to be calculated beforehand, for example, speed.


def check_observations_continuity(tr: "FishTracker", n: int = 2):
    """
    Check that the last N observations have consecutive frame indices.

    Args:
        tr: FishTracker object with metrics list
        n: Number of last observations to check for continuity (default: 2)

    Returns:
        bool: True if last N observations are continuous, False otherwise
    """
    # Need at least N observations to check
    if len(tr.metrics) < n:
        return False

    # Check each consecutive pair in the last N observations
    for i in range(-n, -1):
        prev_metric = tr.metrics[i]
        curr_metric = tr.metrics[i + 1]
        if prev_metric.frame_idx + 1 != curr_metric.frame_idx:
            return False

    return True


class AnomalyDict(TypedDict):
    type: str
    value: float


class FishAnomalyRule(Protocol):
    """
    Protocol defining the interface for fish anomaly detection.

    An anomaly rule is a callable that decides whether an anomaly has occurred within the data tracked by a fish tracker
    """

    def __call__(self, tr: "FishTracker") -> Optional[AnomalyDict]: ...

    def explain(self, anomaly: AnomalyDict) -> str:
        """Return a human-readable explanation of the anomaly"""
        ...


@dataclass
class AreaChangeAnomaly:
    """
    Check for sudden area changes in the fish mask.
    """

    area_change_thresh: float = 0.5

    def __call__(self, tr: "FishTracker") -> Optional[AnomalyDict]:
        """Calculate percentage change in area between the last two frames"""

        if not check_observations_continuity(tr):
            return

        prev_area = tr.metrics[-2].area
        curr_area = tr.metrics[-1].area

        if prev_area > 0:
            area_change = (curr_area - prev_area) / prev_area
        else:
            area_change = 0

        if abs(area_change) > self.area_change_thresh:
            return {"type": "area_spike", "value": round(area_change, 2)}

    def explain(self, anomaly: AnomalyDict) -> str:
        percent_change = anomaly["value"]
        return f"Area changed by {percent_change:.1%} (threshold: {self.area_change_thresh:.1%})"


@dataclass
class LargeDisplacementAnomaly:
    """Check for sudden large movements in fish centroid position"""

    displacement_thresh: int = 300  # pixels

    def __call__(self, tr: "FishTracker") -> Optional[AnomalyDict]:
        """
        Detect anomalously large displacements between consecutive frames.
        """

        if not check_observations_continuity(tr):
            return

        centroids_x = [m.centroid_x for m in tr.metrics]
        centroids_y = [m.centroid_y for m in tr.metrics]

        dx = centroids_x[-1] - centroids_x[-2]
        dy = centroids_y[-1] - centroids_y[-2]
        displacement = np.sqrt(dx**2 + dy**2)

        if displacement > self.displacement_thresh:
            return {"type": "large_displacement", "value": round(displacement, 0)}

    def explain(self, anomaly: AnomalyDict) -> str:
        displacement = anomaly["value"]
        return f"Fish displaced {displacement:.0f} px (threshold: {self.displacement_thresh} px)"


@dataclass
class SolidityChangeAnomaly:
    """
    Flags sudden frame-to-frame jumps in solidity (area / convex hull area).
    Useful to catch merged masks, broken masks, or sharp shape irregularities.

    delta_thresh:
        Relative change threshold. E.g., 0.5 means > ±50% change triggers.
    """

    delta_thresh: float = 0.5  # > ±50% jump

    def __call__(self, tr: "FishTracker") -> Optional[AnomalyDict]:
        if not check_observations_continuity(tr):
            return

        s_curr = tr.metrics[-1].solidity
        s_prev = tr.metrics[-2].solidity

        delta = (s_curr - s_prev) / s_prev
        if abs(delta) > self.delta_thresh:
            return {"type": "solidity_spike", "value": round(delta, 2)}

        return None

    def explain(self, anomaly: AnomalyDict) -> str:
        change = anomaly["value"]
        return (
            f"Solidity changed by {change:+.1%} (threshold: {self.delta_thresh:.0%})."
        )


@dataclass
class CompactnessChangeAnomaly:
    """
    Flags sudden frame-to-frame jumps in solidity (area / convex hull area).
    Useful to catch merged masks, broken masks, or sharp shape irregularities.

    delta_thresh:
        Relative change threshold. E.g., 0.5 means > ±50% change triggers.
    """

    delta_thresh: float = 0.7

    def __call__(self, tr: "FishTracker") -> Optional[AnomalyDict]:
        if not check_observations_continuity(tr):
            return

        s_curr = tr.metrics[-1].compactness
        s_prev = tr.metrics[-2].compactness

        delta = (s_curr - s_prev) / s_prev
        if abs(delta) > self.delta_thresh:
            return {"type": "compactness_spike", "value": round(delta, 2)}

        return None

    def explain(self, anomaly: AnomalyDict) -> str:
        change = anomaly["value"]
        return f"Compactness changed by {change:+.1%} (threshold: {self.delta_thresh:.0%})."


@dataclass
class AreaZScoreAnomaly:
    """Area outlier vs rolling window (robust to gradual drift)."""

    window: int = 6
    z_thresh: float = 3.5

    def __call__(self, tr: "FishTracker"):
        if len(tr.metrics) < self.window:
            return

        xs = [tr.metrics[i].area for i in range(-self.window, -1)]
        mu = float(np.mean(xs))
        sd = float(np.std(xs))
        self.mu = mu
        self.sd = sd
        z = (tr.metrics[-1].area - mu) / sd

        if z > self.z_thresh:
            return {"type": "area_zscore", "value": round(z, 2)}

    def explain(self, anomaly: AnomalyDict) -> str:
        z_score = anomaly["value"]
        direction = "larger" if z_score > 0 else "smaller"
        return f"Fish area is {direction} than expected (z-score: {z_score:.2f}, threshold: {self.z_thresh}), mu: {self.mu}, sd: {self.sd}"
