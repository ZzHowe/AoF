from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from aof_pipeline.providers import BaseProvider, build_provider
from aof_pipeline.types import (
    BoundingBox,
    ConsensusAnnotation,
    ExpertAnnotation,
    FinalSample,
    PatchAssessment,
    ReasoningCandidate,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _image_metrics(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    grad_y, grad_x = np.gradient(gray)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(np.sqrt(np.square(grad_x) + np.square(grad_y)).mean()),
        "saturation": float((rgb.max(axis=2) - rgb.min(axis=2)).mean()),
    }


def quality_filter_score(image: Image.Image) -> float:
    metrics = _image_metrics(image)
    exposure = 1.0 - abs(metrics["brightness"] - 0.5) * 2.0
    contrast = min(metrics["contrast"] / 0.30, 1.0)
    sharpness = min(metrics["sharpness"] / 0.25, 1.0)
    saturation = min(metrics["saturation"] / 0.35, 1.0)
    return round(max(0.0, min(5.0, 5.0 * (0.30 * exposure + 0.30 * contrast + 0.25 * sharpness + 0.15 * saturation))), 3)


def split_box(box: BoundingBox) -> list[BoundingBox]:
    mid_x = (box.x1 + box.x2) // 2
    mid_y = (box.y1 + box.y2) // 2
    return [
        BoundingBox(box.x1, box.y1, mid_x, mid_y),
        BoundingBox(mid_x, box.y1, box.x2, mid_y),
        BoundingBox(box.x1, mid_y, mid_x, box.y2),
        BoundingBox(mid_x, mid_y, box.x2, box.y2),
    ]


def crop_box(image: Image.Image, box: BoundingBox) -> Image.Image:
    return image.crop(box.to_tuple())


def can_split(box: BoundingBox, depth: int, max_depth: int, min_patch_size: int) -> bool:
    return depth < max_depth and min(box.width, box.height) >= 2 * min_patch_size


def collect_patches(
    image: Image.Image,
    image_name: str,
    box: BoundingBox,
    depth: int,
    max_depth: int,
    min_patch_size: int,
    expert: BaseProvider,
    global_semantics: list[str],
) -> list[PatchAssessment]:
    patch = crop_box(image, box)
    assessment = expert.assess_patch(
        patch=patch,
        bbox=box,
        image_name=image_name,
        global_semantics=global_semantics,
    )
    overlap = [item for item in assessment.semantics if item in global_semantics]
    if overlap:
        assessment.semantics = overlap[:4]
    if overlap or not can_split(box, depth, max_depth, min_patch_size):
        if not assessment.semantics:
            assessment.semantics = global_semantics[:1] or ["generic_region"]
        return [assessment]

    patches: list[PatchAssessment] = []
    for child in split_box(box):
        if child.width <= 0 or child.height <= 0:
            continue
        patches.extend(
            collect_patches(
                image=image,
                image_name=image_name,
                box=child,
                depth=depth + 1,
                max_depth=max_depth,
                min_patch_size=min_patch_size,
                expert=expert,
                global_semantics=global_semantics,
            )
        )
    return patches or [assessment]


def aggregate_locations(patches: Iterable[PatchAssessment]) -> dict[str, BoundingBox]:
    locations: dict[str, BoundingBox] = {}
    for patch in patches:
        for semantic in patch.semantics:
            if semantic not in locations:
                locations[semantic] = patch.bbox
            else:
                locations[semantic] = locations[semantic].union(patch.bbox)
    return locations


def run_expert_annotation(
    image_path: Path,
    expert: BaseProvider,
    max_depth: int,
    min_patch_size: int,
) -> ExpertAnnotation:
    image = Image.open(image_path).convert("RGB")
    global_semantics, global_summary = expert.describe_global(image=image, image_name=image_path.name)
    root = BoundingBox(0, 0, image.width, image.height)
    boxes = split_box(root)
    patches: list[PatchAssessment] = []
    for box in boxes:
        patches.extend(
            collect_patches(
                image=image,
                image_name=image_path.name,
                box=box,
                depth=1,
                max_depth=max_depth,
                min_patch_size=min_patch_size,
                expert=expert,
                global_semantics=global_semantics,
            )
        )
    if not patches:
        patches = collect_patches(
            image=image,
            image_name=image_path.name,
            box=root,
            depth=0,
            max_depth=max_depth,
            min_patch_size=min_patch_size,
            expert=expert,
            global_semantics=global_semantics,
        )

    mos = expert.predict_mos(image=image, image_name=image_path.name, global_semantics=global_semantics, patches=patches)
    return ExpertAnnotation(
        expert_name=expert.name,
        global_semantics=global_semantics,
        global_summary=global_summary,
        locations=aggregate_locations(patches),
        patches=patches,
        mos=mos,
        filter_score=quality_filter_score(image),
    )


def intersect_locations(annotations: list[ExpertAnnotation]) -> dict[str, BoundingBox]:
    if not annotations:
        return {}
    common = set(annotations[0].locations)
    for annotation in annotations[1:]:
        common &= set(annotation.locations)

    merged: dict[str, BoundingBox] = {}
    for semantic in sorted(common):
        boxes = [annotation.locations[semantic] for annotation in annotations]
        current = boxes[0]
        for candidate in boxes[1:]:
            intersection = current.intersection(candidate)
            current = intersection if intersection is not None else current.mean_with([candidate])
        merged[semantic] = current
    return merged


def majority_vote(
    image_id: str,
    image_path: Path,
    annotations: list[ExpertAnnotation],
    beta: float,
    min_experts: int,
) -> Optional[ConsensusAnnotation]:
    if not annotations:
        return None
    sorted_annotations = sorted(annotations, key=lambda item: item.mos)
    best_window: list[ExpertAnnotation] = []
    left = 0
    for right in range(len(sorted_annotations)):
        while sorted_annotations[right].mos - sorted_annotations[left].mos >= beta:
            left += 1
        window = sorted_annotations[left : right + 1]
        if len(window) > len(best_window):
            best_window = window

    if len(best_window) < min_experts:
        return None

    common_semantics = set(best_window[0].global_semantics)
    for annotation in best_window[1:]:
        common_semantics &= set(annotation.global_semantics)

    consensus_mos = round(sum(annotation.mos for annotation in best_window) / len(best_window), 3)
    reference = min(best_window, key=lambda item: abs(item.mos - consensus_mos))
    return ConsensusAnnotation(
        image_id=image_id,
        image_path=str(image_path),
        mos=consensus_mos,
        accepted_experts=[annotation.expert_name for annotation in best_window],
        global_semantics=sorted(common_semantics) or reference.global_semantics[:4],
        locations=intersect_locations(best_window) or reference.locations,
        reference_patches=reference.patches,
        debug_votes={annotation.expert_name: annotation.mos for annotation in annotations},
    )


def reject_sample(
    image: Image.Image,
    consensus: ConsensusAnnotation,
    reasoning_provider: BaseProvider,
    judge_provider: BaseProvider,
    rollouts: int,
    threshold: float,
) -> Optional[ReasoningCandidate]:
    best: Optional[ReasoningCandidate] = None
    for rollout_id in range(rollouts):
        reasoning_text, predicted_mos = reasoning_provider.generate_reasoning_rollout(
            image=image,
            consensus=consensus,
            rollout_id=rollout_id,
        )
        judge_score = judge_provider.judge_reasoning(
            image=image,
            consensus=consensus,
            reasoning_text=reasoning_text,
            predicted_mos=predicted_mos,
        )
        candidate = ReasoningCandidate(
            reasoning=reasoning_text,
            predicted_mos=predicted_mos,
            judge_score=judge_score,
            accepted=judge_score >= threshold,
        )
        if candidate.accepted:
            return candidate
        if best is None or candidate.judge_score > best.judge_score:
            best = candidate
    return best if best and best.accepted else None


def serialize_sample(sample: FinalSample) -> str:
    return json.dumps(sample.to_dict(), ensure_ascii=False)


def list_images(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def run_pipeline(config: dict[str, Any], config_dir: Path) -> dict[str, int]:
    paths_cfg = config["paths"]
    pipeline_cfg = config["pipeline"]
    provider_cfg = config["providers"]

    raw_dir = resolve_path(config_dir, paths_cfg["raw_dir"])
    output_dir = _ensure_dir(resolve_path(config_dir, paths_cfg["output_dir"]))
    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"

    experts = [build_provider(item) for item in provider_cfg["experts"]]
    reasoning_provider = build_provider(provider_cfg["reasoning"])
    judge_provider = build_provider(provider_cfg["judge"])

    images = list_images(raw_dir)
    accepted_lines: list[str] = []
    rejected_lines: list[str] = []
    summary = {"total": len(images), "accepted": 0, "rejected": 0, "filtered": 0}

    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        filter_score = quality_filter_score(image)
        if filter_score < float(pipeline_cfg["min_filter_score"]) or filter_score > float(pipeline_cfg["max_filter_score"]):
            summary["filtered"] += 1
            summary["rejected"] += 1
            rejected_lines.append(
                json.dumps(
                    {
                        "image_id": image_path.stem,
                        "image_path": str(image_path),
                        "reason": "coarse_filter",
                        "filter_score": filter_score,
                    },
                    ensure_ascii=False,
                )
            )
            continue

        annotations = [
            run_expert_annotation(
                image_path=image_path,
                expert=expert,
                max_depth=int(pipeline_cfg["max_depth"]),
                min_patch_size=int(pipeline_cfg["min_patch_size"]),
            )
            for expert in experts
        ]
        consensus = majority_vote(
            image_id=image_path.stem,
            image_path=image_path,
            annotations=annotations,
            beta=float(pipeline_cfg["vote_beta"]),
            min_experts=int(pipeline_cfg["min_experts"]),
        )
        if consensus is None:
            summary["rejected"] += 1
            rejected_lines.append(
                json.dumps(
                    {
                        "image_id": image_path.stem,
                        "image_path": str(image_path),
                        "reason": "majority_vote_failed",
                        "votes": {item.expert_name: item.mos for item in annotations},
                    },
                    ensure_ascii=False,
                )
            )
            continue

        reasoning = reject_sample(
            image=image,
            consensus=consensus,
            reasoning_provider=reasoning_provider,
            judge_provider=judge_provider,
            rollouts=int(pipeline_cfg["reject_rollouts"]),
            threshold=float(pipeline_cfg["judge_accept_threshold"]),
        )
        if reasoning is None:
            summary["rejected"] += 1
            rejected_lines.append(
                json.dumps(
                    {
                        "image_id": image_path.stem,
                        "image_path": str(image_path),
                        "reason": "reject_sampling_failed",
                        "consensus_mos": consensus.mos,
                    },
                    ensure_ascii=False,
                )
            )
            continue

        sample = FinalSample(
            image_id=image_path.stem,
            image_path=str(image_path),
            mos=consensus.mos,
            thinking=reasoning.reasoning,
            location={key: value.to_tuple() for key, value in consensus.locations.items()},
            global_semantics=consensus.global_semantics,
            accepted_experts=consensus.accepted_experts,
            debug_votes=consensus.debug_votes,
        )
        summary["accepted"] += 1
        accepted_lines.append(serialize_sample(sample))

    accepted_path.write_text("\n".join(accepted_lines) + ("\n" if accepted_lines else ""), encoding="utf-8")
    rejected_path.write_text("\n".join(rejected_lines) + ("\n" if rejected_lines else ""), encoding="utf-8")
    return summary


def create_demo_dataset(output_dir: Path, count: int = 12, seed: int = 7) -> list[Path]:
    rng = random.Random(seed)
    output_dir = _ensure_dir(output_dir)
    generated: list[Path] = []
    labels = ["street", "forest", "city", "portrait", "coast", "mountain"]

    for idx in range(count):
        width = 512
        height = 512
        image = Image.new("RGB", (width, height), color=(rng.randint(20, 180), rng.randint(20, 180), rng.randint(20, 180)))
        draw = ImageDraw.Draw(image)
        for _ in range(14):
            x1 = rng.randint(0, width - 80)
            y1 = rng.randint(0, height - 80)
            x2 = x1 + rng.randint(40, 220)
            y2 = y1 + rng.randint(40, 220)
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            if rng.random() > 0.5:
                draw.rectangle((x1, y1, x2, y2), fill=color, outline=None)
            else:
                draw.ellipse((x1, y1, x2, y2), fill=color, outline=None)

        distortions: list[str] = []
        if idx % 3 == 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.2, 2.8)))
            distortions.append("blur")
        if idx % 4 == 0:
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.45, 0.75))
            distortions.append("lowcontrast")
        if idx % 5 == 0:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.45, 0.72))
            distortions.append("dark")
        if idx % 2 == 0:
            array = np.asarray(image, dtype=np.int16)
            noise = rng.randint(8, 28)
            noisy = np.clip(array + np.random.default_rng(seed + idx).normal(0, noise, array.shape), 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy)
            distortions.append("noise")

        name = f"{idx:03d}_{labels[idx % len(labels)]}_{'_'.join(distortions) or 'clean'}.png"
        path = output_dir / name
        image.save(path)
        generated.append(path)
    return generated
