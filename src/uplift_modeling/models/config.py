"""Resolve shared model defaults and candidate-specific overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uplift_modeling.models.scoring import SUPPORTED_MODEL_KINDS
from uplift_modeling.utils.config import get_config_section


@dataclass(frozen=True)
class ModelCandidateConfig:
    """Resolved configuration for one model candidate."""

    name: str
    kind: str
    params: dict[str, Any]


def resolve_model_candidates(
    config: dict[str, Any],
) -> tuple[ModelCandidateConfig, ...]:
    """Resolve model defaults and candidate overrides."""
    models_config = get_config_section(config, "models")

    model_defaults = models_config.get("model_defaults", {})
    if model_defaults is None:
        model_defaults = {}

    if not isinstance(model_defaults, dict):
        raise ValueError("models.model_defaults must be a mapping.")

    raw_candidates = models_config.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("models.candidates must be a non-empty list.")

    resolved_candidates: list[ModelCandidateConfig] = []
    seen_names: set[str] = set()

    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            raise ValueError(
                f"models.candidates[{index}] must be a mapping."
            )

        name = raw_candidate.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"models.candidates[{index}].name must be a non-empty string."
            )
        name = name.strip()

        kind = raw_candidate.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(
                f"models.candidates[{index}].kind must be a non-empty string."
            )
        kind = kind.strip()

        if kind not in SUPPORTED_MODEL_KINDS:
            supported = ", ".join(SUPPORTED_MODEL_KINDS)
            raise ValueError(
                f"Unsupported model kind '{kind}'. "
                f"Supported kinds: {supported}."
            )

        if name in seen_names:
            raise ValueError(f"Duplicate candidate model name: {name}")
        seen_names.add(name)

        candidate_params = raw_candidate.get("params", {})
        if candidate_params is None:
            candidate_params = {}

        if not isinstance(candidate_params, dict):
            raise ValueError(
                f"models.candidates[{index}].params must be a mapping."
            )

        resolved_params = {
            **model_defaults,
            **candidate_params,
        }

        resolved_candidates.append(
            ModelCandidateConfig(
                name=name,
                kind=kind,
                params=resolved_params,
            )
        )

    return tuple(resolved_candidates)


def resolve_model_candidate(
    config: dict[str, Any],
    model_kind: str,
) -> ModelCandidateConfig:
    """Resolve the single configured candidate for one model kind."""
    candidates = resolve_model_candidates(config)

    matches = [
        candidate
        for candidate in candidates
        if candidate.kind == model_kind
    ]

    if not matches:
        raise ValueError(
            f"No model candidate configured for kind '{model_kind}'."
        )

    if len(matches) > 1:
        names = [candidate.name for candidate in matches]
        raise ValueError(
            f"Multiple model candidates configured for kind '{model_kind}': "
            f"{names}."
        )

    return matches[0]