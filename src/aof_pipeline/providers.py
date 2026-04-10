from __future__ import annotations

import base64
import io
import json
import math
import os
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import requests
from PIL import Image

from aof_pipeline.types import BoundingBox, ConsensusAnnotation, PatchAssessment


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _tokenize_filename(image_name: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]+", Path(image_name).stem.lower())
    return [token for token in tokens if len(token) > 2]


def _image_metrics(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    brightness = float(gray.mean())
    contrast = float(gray.std())
    grad_y, grad_x = np.gradient(gray)
    sharpness = float(np.sqrt(np.square(grad_x) + np.square(grad_y)).mean())
    saturation = float((rgb.max(axis=2) - rgb.min(axis=2)).mean())
    colorfulness = float(np.abs(rgb[:, :, 0] - rgb[:, :, 1]).mean() + np.abs(rgb[:, :, 1] - rgb[:, :, 2]).mean())
    noise = float(np.abs(gray - gray.mean()).mean())
    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "saturation": saturation,
        "colorfulness": colorfulness,
        "noise": noise,
    }


class BaseProvider(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def describe_global(self, image: Image.Image, image_name: str) -> tuple[list[str], str]:
        raise NotImplementedError

    @abstractmethod
    def assess_patch(
        self,
        patch: Image.Image,
        bbox: BoundingBox,
        image_name: str,
        global_semantics: Sequence[str],
    ) -> PatchAssessment:
        raise NotImplementedError

    @abstractmethod
    def predict_mos(
        self,
        image: Image.Image,
        image_name: str,
        global_semantics: Sequence[str],
        patches: Sequence[PatchAssessment],
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def generate_reasoning_rollout(
        self,
        image: Image.Image,
        consensus: ConsensusAnnotation,
        rollout_id: int,
    ) -> tuple[str, float]:
        raise NotImplementedError

    @abstractmethod
    def judge_reasoning(
        self,
        image: Image.Image,
        consensus: ConsensusAnnotation,
        reasoning_text: str,
        predicted_mos: float,
    ) -> float:
        raise NotImplementedError


class MockProvider(BaseProvider):
    def __init__(
        self,
        name: str,
        seed: int = 0,
        hallucination_rate: float = 0.18,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(name=name)
        self.seed = seed
        self.hallucination_rate = hallucination_rate
        self.temperature = temperature

    def _rng(self, *parts: Any) -> random.Random:
        return random.Random("|".join(str(part) for part in (self.seed, self.name, *parts)))

    def _semantic_labels(self, image: Image.Image, image_name: str) -> list[str]:
        metrics = _image_metrics(image)
        labels: list[str] = []
        labels.extend(_tokenize_filename(image_name)[:3])
        brightness = metrics["brightness"]
        contrast = metrics["contrast"]
        sharpness = metrics["sharpness"]
        colorfulness = metrics["colorfulness"]

        labels.append("bright" if brightness > 0.6 else "dark" if brightness < 0.4 else "balanced_lighting")
        labels.append("textured" if sharpness > 0.12 else "smooth")
        labels.append("high_contrast" if contrast > 0.22 else "low_contrast")
        labels.append("colorful" if colorfulness > 0.22 else "muted")

        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        dominant = int(np.argmax(rgb))
        labels.append(["reddish", "greenish", "bluish"][dominant])
        return list(dict.fromkeys(labels))

    def _quality_score(self, metrics: dict[str, float], rng: Optional[random.Random] = None) -> float:
        exposure = 1.0 - abs(metrics["brightness"] - 0.5) * 2.0
        contrast = _clip(metrics["contrast"] / 0.30, 0.0, 1.0)
        sharpness = _clip(metrics["sharpness"] / 0.25, 0.0, 1.0)
        saturation = _clip(metrics["saturation"] / 0.35, 0.0, 1.0)
        noise_penalty = _clip(metrics["noise"] / 0.30, 0.0, 1.0)
        score = 5.0 * (0.28 * exposure + 0.28 * contrast + 0.28 * sharpness + 0.16 * saturation)
        score = score - 0.8 * max(0.0, noise_penalty - 0.45)
        if rng is not None:
            score += rng.uniform(-0.08, 0.08)
        return round(_clip(score, 0.0, 5.0), 3)

    def _quality_summary(self, metrics: dict[str, float]) -> str:
        reasons: list[str] = []
        if metrics["sharpness"] < 0.08:
            reasons.append("blurred texture")
        if metrics["contrast"] < 0.12:
            reasons.append("weak local contrast")
        if metrics["brightness"] < 0.28:
            reasons.append("under-exposed content")
        elif metrics["brightness"] > 0.72:
            reasons.append("over-exposed highlight")
        if metrics["saturation"] < 0.08:
            reasons.append("desaturated color")
        if not reasons:
            reasons.append("stable structure and clear details")
        return ", ".join(reasons[:3])

    def describe_global(self, image: Image.Image, image_name: str) -> tuple[list[str], str]:
        semantics = self._semantic_labels(image, image_name)
        metrics = _image_metrics(image)
        summary = (
            f"Global scene hints: {', '.join(semantics[:5])}. "
            f"Brightness={metrics['brightness']:.2f}, contrast={metrics['contrast']:.2f}, sharpness={metrics['sharpness']:.2f}."
        )
        return semantics, summary

    def assess_patch(
        self,
        patch: Image.Image,
        bbox: BoundingBox,
        image_name: str,
        global_semantics: Sequence[str],
    ) -> PatchAssessment:
        rng = self._rng(image_name, bbox.to_tuple())
        metrics = _image_metrics(patch)
        local_semantics = self._semantic_labels(patch, image_name)
        if rng.random() < self.hallucination_rate:
            local_semantics.append(rng.choice(["hallucinated_object", "reflection", "watermark"]))
        local_semantics = list(dict.fromkeys(local_semantics))
        overlap = [item for item in local_semantics if item in global_semantics]
        if overlap:
            local_semantics = overlap + [item for item in local_semantics if item not in overlap][:2]
        score = self._quality_score(metrics, rng)
        summary = self._quality_summary(metrics)
        return PatchAssessment(
            bbox=bbox,
            semantics=local_semantics[:4],
            quality_score=score,
            quality_summary=summary,
            metrics={key: round(value, 4) for key, value in metrics.items()},
        )

    def predict_mos(
        self,
        image: Image.Image,
        image_name: str,
        global_semantics: Sequence[str],
        patches: Sequence[PatchAssessment],
    ) -> float:
        image_metrics = _image_metrics(image)
        global_score = self._quality_score(image_metrics, self._rng(image_name, "global"))
        if not patches:
            return global_score
        weighted = sum(patch.quality_score * max(1, patch.bbox.area) for patch in patches)
        total_area = sum(max(1, patch.bbox.area) for patch in patches)
        patch_score = weighted / total_area
        stability_bonus = 0.1 if "balanced_lighting" in global_semantics else -0.05
        return round(_clip(0.35 * global_score + 0.65 * patch_score + stability_bonus, 0.0, 5.0), 3)

    def generate_reasoning_rollout(
        self,
        image: Image.Image,
        consensus: ConsensusAnnotation,
        rollout_id: int,
    ) -> tuple[str, float]:
        rng = self._rng(consensus.image_id, "rollout", rollout_id)
        patches = sorted(consensus.reference_patches, key=lambda item: item.quality_score)[:3]
        steps = [
            f"Step 1: The global semantics are {', '.join(consensus.global_semantics[:4]) or 'generic_scene'}.",
            "Step 2: Inspect the most quality-sensitive local regions.",
        ]
        for idx, patch in enumerate(patches):
            loc_name = f"loc{idx}"
            steps.append(
                f"- {patch.bbox.xml_tag(loc_name)} region shows {patch.quality_summary}; "
                f"local score={patch.quality_score:.2f} and semantics={', '.join(patch.semantics[:3])}."
            )
        predicted = round(_clip(consensus.mos + rng.uniform(-0.12, 0.12), 0.0, 5.0), 3)
        steps.append(
            f"Step 3: Aggregate the local quality evidence and output the final MOS={predicted:.3f}/5.0."
        )
        return "\n".join(steps), predicted

    def judge_reasoning(
        self,
        image: Image.Image,
        consensus: ConsensusAnnotation,
        reasoning_text: str,
        predicted_mos: float,
    ) -> float:
        distance = abs(predicted_mos - consensus.mos)
        score = 1.0 - distance / 0.5
        mentions_loc = 0.1 if "<loc0>" in reasoning_text or "<loc>" in reasoning_text else 0.0
        mentions_semantics = 0.0
        if any(label in reasoning_text for label in consensus.global_semantics[:3]):
            mentions_semantics = 0.1
        return round(_clip(score + mentions_loc + mentions_semantics, 0.0, 1.0), 3)


class OpenAICompatibleProvider(BaseProvider):
    def __init__(
        self,
        name: str,
        model: str,
        api_base: str,
        api_key_env: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> None:
        super().__init__(name=name)
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _api_key(self) -> str:
        api_key = os.getenv(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        return api_key

    def _image_url(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _chat_json(self, image: Image.Image, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": self._image_url(image)}},
                    ],
                },
            ],
        }
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        return self._extract_json(content)

    def describe_global(self, image: Image.Image, image_name: str) -> tuple[list[str], str]:
        result = self._chat_json(
            image=image,
            system_prompt="You are an image-quality expert. Output compact JSON only.",
            user_prompt=(
                "Return JSON with keys `semantics` (list of 3-8 scene labels) and `summary` "
                "(one sentence global scene summary). Focus on scene semantics, not quality."
            ),
        )
        semantics = [str(item).strip().lower().replace(" ", "_") for item in result.get("semantics", [])]
        return semantics[:8], str(result.get("summary", "")).strip()

    def assess_patch(
        self,
        patch: Image.Image,
        bbox: BoundingBox,
        image_name: str,
        global_semantics: Sequence[str],
    ) -> PatchAssessment:
        result = self._chat_json(
            image=patch,
            system_prompt="You are an image-quality expert. Output compact JSON only.",
            user_prompt=(
                "Return JSON with keys `semantics` (local labels), `quality_score` (0-5 float), "
                "`quality_summary` (one sentence). Use the global scene hints: "
                f"{', '.join(global_semantics[:6]) or 'none'}."
            ),
        )
        semantics = [str(item).strip().lower().replace(" ", "_") for item in result.get("semantics", [])]
        return PatchAssessment(
            bbox=bbox,
            semantics=semantics[:4],
            quality_score=round(float(result.get("quality_score", 2.5)), 3),
            quality_summary=str(result.get("quality_summary", "")).strip(),
            metrics={},
        )

    def predict_mos(
        self,
        image: Image.Image,
        image_name: str,
        global_semantics: Sequence[str],
        patches: Sequence[PatchAssessment],
    ) -> float:
        patch_text = "\n".join(
            f"- bbox={patch.bbox.to_tuple()}, score={patch.quality_score:.3f}, summary={patch.quality_summary}"
            for patch in patches[:12]
        )
        result = self._chat_json(
            image=image,
            system_prompt="You are an image-quality expert. Output compact JSON only.",
            user_prompt=(
                "Predict the final MOS from 0 to 5. Return JSON with keys `mos` and `rationale`. "
                f"Global semantics: {', '.join(global_semantics[:8])}. Local evidence:\n{patch_text}"
            ),
        )
        return round(float(result.get("mos", 2.5)), 3)

    def generate_reasoning_rollout(
        self,
        image: Image.Image,
        consensus: ConsensusAnnotation,
        rollout_id: int,
    ) -> tuple[str, float]:
        patch_text = "\n".join(
            f"- bbox={patch.bbox.to_tuple()}, score={patch.quality_score:.3f}, summary={patch.quality_summary}"
            for patch in consensus.reference_patches[:8]
        )
        result = self._chat_json(
            image=image,
            system_prompt="You are a professional image-quality-assessment expert. Output compact JSON only.",
            user_prompt=(
                "Perform step-wise semantic-position reasoning. Return JSON with keys `reasoning` and "
                "`predicted_mos`. Use XML-like location tags such as <loc0>(x1, y1, x2, y2)</loc0> in the reasoning. "
                f"Global semantics: {', '.join(consensus.global_semantics[:8])}. Supporting local evidence:\n{patch_text}"
            ),
        )
        return str(result.get("reasoning", "")).strip(), round(float(result.get("predicted_mos", 2.5)), 3)

    def judge_reasoning(
        self,
        image: Image.Image,
        consensus: ConsensusAnnotation,
        reasoning_text: str,
        predicted_mos: float,
    ) -> float:
        result = self._chat_json(
            image=image,
            system_prompt="You are a strict judge for image-quality reasoning. Output compact JSON only.",
            user_prompt=(
                "Return JSON with keys `judge_score` (0-1 float) and `verdict`. "
                f"Target consensus MOS={consensus.mos:.3f}; predicted MOS={predicted_mos:.3f}. "
                f"Reasoning text:\n{reasoning_text}"
            ),
        )
        return round(float(result.get("judge_score", 0.0)), 3)


def build_provider(config: dict[str, Any]) -> BaseProvider:
    provider_type = config.get("type", "mock")
    name = config.get("name", provider_type)
    if provider_type == "mock":
        return MockProvider(
            name=name,
            seed=int(config.get("seed", 0)),
            hallucination_rate=float(config.get("hallucination_rate", 0.18)),
            temperature=float(config.get("temperature", 0.2)),
        )
    if provider_type == "openai_compatible":
        return OpenAICompatibleProvider(
            name=name,
            model=str(config["model"]),
            api_base=str(config["api_base"]),
            api_key_env=str(config["api_key_env"]),
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 1200)),
        )
    raise ValueError(f"Unsupported provider type: {provider_type}")
