# Standard Library imports
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

"""
Types copied and modified from:
labelme_types.py at https://github.com/mad4octos/LabelMe/
"""

@dataclass
class CocoRLE:
    counts: list[int]
    size: list[int]

    @classmethod
    def from_dict(cls, d: dict) -> "CocoRLE":
        return cls(counts=d["counts"], size=d["size"])


@dataclass
class CocoImage:
    id: int
    filepath: Path
    width: int
    height: int

    @classmethod
    def from_dict(cls, d: dict) -> "CocoImage":
        return cls(
            id=d["id"],
            filepath=Path(d["file_name"]),
            width=d["width"],
            height=d["height"],
        )


# A flat list of coordinates in COCO polygon format: [x1, y1, x2, y2, ..., xn, yn]
CocoPolygon = list[float]


@dataclass
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int

    # Polygon segmentation (iscrowd == 0) OR RLE (iscrowd == 1)
    segmentation: list[CocoPolygon] | CocoRLE

    area: float
    bbox: list[float]  # [x, y, width, height]

    # From the COCO specification:
    # https://cocodataset.org/#format-data
    #
    # The segmentation format depends on whether the instance represents
    # a single object (iscrowd=0 in which case polygons are used) or a
    # collection of objects (iscrowd=1 in which case RLE is used).
    # Note that a single object (iscrowd=0) may require multiple polygons,
    # for example if occluded. Crowd annotations (iscrowd=1) are used to label large
    # groups of objects (e.g. a crowd of people).
    iscrowd: Literal[0, 1]

    # Optional, non-standard COCO field
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "CocoAnnotation":
        return cls(
            id=d["id"],
            image_id=d["image_id"],
            category_id=d["category_id"],
            segmentation=CocoRLE.from_dict(d["segmentation"])
            if isinstance(d["segmentation"], dict)
            else d["segmentation"],
            area=d["area"],
            bbox=d["bbox"],
            iscrowd=d["iscrowd"],
            attributes=d.get("attributes", {}),
        )


class CocoCategories(TypedDict):
    id: int
    name: str
    supercategory: str


class CocoFile(TypedDict):
    images: list[CocoImage]
    categories: list[CocoCategories]
    annotations: list[CocoAnnotation]
