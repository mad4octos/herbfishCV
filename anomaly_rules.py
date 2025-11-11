# Standard Library imports
from dataclasses import dataclass
from typing import Optional, TypedDict, Protocol

# External imports
import numpy as np
import pandas as pd

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
class ZScoreAnomaly:
    """
    This class is used to track properties of a blob, assuming they are normally distributed, and rejecting observations
    when they fall out of certain number of standard deviations.

    """

    metric_name: str
    window: int = 6
    z_thresh: float = 3.5
    detrend = True
    smooth = True

    def __call__(self, tr: "FishTracker"):
        """ """

        if len(tr.metrics) < self.window:
            return

        xs = [getattr(tr.metrics[i], self.metric_name) for i in range(-self.window, 0)]
        if self.detrend:
            xs = np.diff(xs)
        if self.smooth:
            xs = pd.Series(xs).ewm(span=self.window).mean().to_numpy()

        self.mu = float(np.mean(xs[:-1]))
        self.sd = float(np.std(xs[:-1]))
        last_metric = xs[-1]
        z = (last_metric - self.mu) / self.sd

        if z > self.z_thresh:
            return {"type": f"{self.metric_name}_zscore", "value": round(z, 2)}

    def explain(self, anomaly: AnomalyDict) -> str:
        z_score = anomaly["value"]
        direction = "larger" if z_score > 0 else "smaller"
        return f"Fish {self.metric_name} is {direction} than expected (z-score: {z_score:.2f}, threshold: {self.z_thresh:.2f}), mu: {self.mu:.2f}, sd: {self.sd:.2f}"


@dataclass
class SpikeAnomaly:
    """
    Check for sudden changes in the fish mask.
    """

    metric_name: str
    change_thresh: float = 0.5

    def __call__(self, tr: "FishTracker") -> Optional[AnomalyDict]:
        """Calculate percentage change in a metric between the last two frames"""

        if not check_observations_continuity(tr):
            return

        prev = getattr(tr.metrics[-2], self.metric_name)
        curr = getattr(tr.metrics[-1], self.metric_name)

        if prev > 0:
            change = (curr - prev) / prev
        else:
            change = 0

        if abs(change) > self.change_thresh:
            return {"type": f"{self.metric_name}_spike", "value": round(change, 2)}

    def explain(self, anomaly: AnomalyDict) -> str:
        percent_change = anomaly["value"]
        return f"{self.metric_name} changed by {percent_change:.1%} (threshold: {self.change_thresh:.1%})"


@dataclass
class AbsoluteThresholdAnomaly:
    """
    Flags if the latest value of `metric_name` violates fixed bounds.
    No dependence on past values.
    """

    metric_name: str
    max_val: Optional[float] = None

    def __call__(self, tr: "FishTracker") -> Optional[AnomalyDict]:
        if not tr.metrics:
            return

        latest_metric_val = float(getattr(tr.metrics[-1], self.metric_name))

        high_violation = self.max_val is not None and (latest_metric_val > self.max_val)

        if high_violation:
            return {
                "type": f"{self.metric_name}_abs_threshold",
                "value": round(float(latest_metric_val), 2),
            }

    def explain(self, anomaly: AnomalyDict) -> str:
        anom_value = anomaly["value"]

        direction = ""
        if self.max_val is not None and (anom_value > self.max_val):
            direction = "above max"
        return (
            f"Absolute threshold violation: {self.metric_name}={anom_value:.2f} is {direction} "
            f"max>'{self.max_val}"
        )
