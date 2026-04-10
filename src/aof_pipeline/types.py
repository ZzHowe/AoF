from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def intersects(self, other: "BoundingBox") -> bool:
        return self.intersection(other) is not None

    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1, y1, x2, y2)

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )

    def mean_with(self, others: list["BoundingBox"]) -> "BoundingBox":
        boxes = [self, *others]
        return BoundingBox(
            x1=round(sum(box.x1 for box in boxes) / len(boxes)),
            y1=round(sum(box.y1 for box in boxes) / len(boxes)),
            x2=round(sum(box.x2 for box in boxes) / len(boxes)),
            y2=round(sum(box.y2 for box in boxes) / len(boxes)),
        )

    def xml_tag(self, name: str = "loc") -> str:
        return f"<{name}>({self.x1}, {self.y1}, {self.x2}, {self.y2})</{name}>"


@dataclass
class PatchAssessment:
    bbox: BoundingBox
    semantics: list[str]
    quality_score: float
    quality_summary: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ExpertAnnotation:
    expert_name: str
    global_semantics: list[str]
    global_summary: str
    locations: dict[str, BoundingBox]
    patches: list[PatchAssessment]
    mos: float
    filter_score: float


@dataclass
class ConsensusAnnotation:
    image_id: str
    image_path: str
    mos: float
    accepted_experts: list[str]
    global_semantics: list[str]
    locations: dict[str, BoundingBox]
    reference_patches: list[PatchAssessment]
    debug_votes: dict[str, float]


@dataclass
class ReasoningCandidate:
    reasoning: str
    predicted_mos: float
    judge_score: float
    accepted: bool


@dataclass
class FinalSample:
    image_id: str
    image_path: str
    mos: float
    thinking: str
    location: dict[str, tuple[int, int, int, int]]
    global_semantics: list[str]
    accepted_experts: list[str]
    debug_votes: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dataclass_to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj
