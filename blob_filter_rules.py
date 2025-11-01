from dataclasses import dataclass
from typing import Protocol


class BlobRule(Protocol):
    """
    Protocol defining the interface for blob filtering rules.

    A blob rule is a callable that decides whether a given blob should be kept or discarded, typically to remove noise
    or irrelevant detections.
    """

    def __call__(self, blob: "BlobInfo") -> bool: ...

    def explain(self, blob: "BlobInfo") -> str: ...


@dataclass
class MinAreaRule:
    min_area: float

    def __call__(self, blob: "BlobInfo") -> bool:
        return blob.area >= self.min_area

    def explain(self, blob: "BlobInfo") -> str:
        return f"area {blob.area} < {self.min_area}"


@dataclass
class MinSizeRule:
    min_size: float

    def __call__(self, blob: "BlobInfo") -> bool:
        return blob.w >= self.min_size and blob.h >= self.min_size

    def explain(self, blob: "BlobInfo") -> str:
        return f"size {blob.w}x{blob.h} < {self.min_size}"
