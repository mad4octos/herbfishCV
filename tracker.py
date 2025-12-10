# Standard Library imports
from collections import deque
from dataclasses import replace
from typing import Iterable, TypedDict

# External imports
import numpy as np

# Local imports
from anomaly_rules import AnomalyDict, FishAnomalyRule
from blob import BlobInfo


class AnomaliesResultsDict(TypedDict):
    cycles_since_update: int
    anomalies: list[AnomalyDict]


class FishTracker:
    def __init__(
        self, obj_id, anomaly_rules: Iterable[FishAnomalyRule], logger, window_size=10
    ):
        self.id = obj_id
        self.window_size = window_size

        # Number of cycles elapsed since the last update.
        # - This counter is incremented during each call to predict().
        # - It resets to 0 whenever update() is called.
        # - The tracker is removed after certain number of cycles without updates
        self.cycles_since_update = 0

        # Bounded deque: discards items from the opposite end when full.
        # Time series of blob properties
        self.metrics: deque[BlobInfo] = deque(maxlen=window_size)

        self.anomaly_rules = anomaly_rules

        self.logger = logger

    def log_metrics(self):
        """Pretty-print metrics deque to the logger."""
        if not self.metrics:
            self.logger.info("No metrics available.")
            return

        lines = ["BlobInfo list (most recent last):"]
        for m in self.metrics:
            lines.append(str(m))

        self.logger.info("\n".join(lines))

    def predict(self) -> AnomaliesResultsDict:
        """Check for anomalies in the tracked fish properties."""
        self.cycles_since_update += 1

        results: AnomaliesResultsDict = {
            "cycles_since_update": self.cycles_since_update,
            "anomalies": [],
        }

        self.log_metrics()
        for anomaly_check in self.anomaly_rules:
            if (anomaly := anomaly_check(self)) is not None:
                # I'm unsure whether to keep or remove the latest metric, both options have trade-offs.
                # If I keep it, the faulty metric skews the baseline and hides future anomalies.
                # If I remove it, the discontinuity makes some anomaly rules be skipped.
                # del self.metrics[-1]

                results["anomalies"].append(anomaly)
                self.logger.warning(anomaly_check.explain(anomaly))

            elif anomaly is None:
                # No anomaly found... but that doesn't guarantee there are no anomalies in this blob.
                # It may just be that there was not enough data to check for anomalies!
                # E.g., for the spike anomaly detector (prev vs current frame),
                # it may be that the previous and current frames are not continuous.
                # That may be due to the classifier rejecting one frame, thus creating a discontinuity between
                # frames. i.e. this anomaly check only acts on continuous non-classifier-rejected frames.
                # The frame could still potentially be anomalous!
                pass

        return results

    def update(self, blob: BlobInfo):
        """ """

        # Filter out old blobs
        current_frame_idx = blob.frame_idx
        filtered: deque[BlobInfo] = deque(maxlen=self.window_size)
        for b in self.metrics:
            if b.frame_idx < (current_frame_idx - self.window_size):
                print(f"Removing old blob {b}")
            else:
                filtered.append(b)
        self.metrics = filtered

        self.cycles_since_update = 0
        # Create a copy with labeled_mask set to None to save memory
        metrics_blob = replace(blob, labeled_mask=None)
        self.metrics.append(metrics_blob)

    def get_summary(self) -> dict:
        """Get current summary of tracked properties."""
        # TODO: Summary of other properties
        areas = [m.area for m in self.metrics]
        return {
            "fish_id": self.id,
            "current_area": areas[-1],
            "area_mean": np.mean(areas),
            "area_variance": np.var(areas) if len(areas) > 1 else 0.0,
            "num_frames_tracked": len(areas),
        }

    def __str__(self) -> str:
        summary = self.get_summary()
        return (
            f"FishTracker(id={self.id}, cycles_since_update={self.cycles_since_update}, "
            f"window_size={self.window_size}, frames_tracked={summary['num_frames_tracked']})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class FishTrackerManager:
    """
    Main tracker that manages multiple FishTracker instances.

    Responsibilities:
    - Create and update individual fish trackers
    - Run predictions on all trackers
    - Remove inactive trackers after a threshold of cycles without updates
    """

    def __init__(
        self,
        anomaly_rules: Iterable[FishAnomalyRule],
        logger,
        max_cycles_without_update=3,
        window_size=10,
    ):
        """ """
        self.trackers: dict[int, FishTracker] = {}
        self.max_cycles_without_update = max_cycles_without_update
        self.window_size = window_size
        self.anomaly_rules = anomaly_rules
        self.logger = logger

    def update(self, blob: BlobInfo):
        """Update a fish's status or create a new fish tracker if not exists."""

        if blob.obj_id not in self.trackers:
            self.trackers[blob.obj_id] = FishTracker(
                blob.obj_id,
                window_size=self.window_size,
                anomaly_rules=self.anomaly_rules,
                logger=self.logger,
            )

        self.trackers[blob.obj_id].update(blob)

    def predict(self, obj_id) -> AnomaliesResultsDict:
        """ """
        return self.trackers[obj_id].predict()

    def filter_dead_trackers(self):
        """ """
        removed_ids = []
        alive_trackers = {}

        for obj_id, tracker in self.trackers.items():
            if tracker.cycles_since_update < self.max_cycles_without_update:
                alive_trackers[obj_id] = tracker
            else:
                removed_ids.append(obj_id)
                print(
                    f"Removing inactive tracker: fish_id={obj_id}, "
                    f"cycles_without_update={tracker.cycles_since_update}"
                )

        self.trackers = alive_trackers
        return removed_ids

    def get_summary(self) -> dict:
        """
        Get a summary of all active trackers.

        Returns:
            Dictionary with tracker statistics
        """
        return {
            "num_active_trackers": len(self.trackers),
            "fish_ids": list(self.trackers.keys()),
            "max_cycles_without_update": self.max_cycles_without_update,
        }

    def __str__(self) -> str:
        summary = self.get_summary()
        return (
            f"FishTrackerManager(active={summary['num_active_trackers']}, "
            f"fish_ids={summary['fish_ids']})"
        )

    def __repr__(self) -> str:
        return self.__str__()
