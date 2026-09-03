"""Auditable label normalization for the governed three-class experiment."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ml_nids.config import LABEL_ALIASES_BY_ROLE, TARGET_LABELS


def validate_role(role: str) -> None:
    """Reject roles that do not have an explicit label policy."""

    if role not in LABEL_ALIASES_BY_ROLE:
        raise ValueError(f"Unsupported dataset role: {role}")


def normalize_label_value(value: object, role: str) -> str:
    """Normalize one source label and apply the role-scoped alias policy."""

    validate_role(role)
    normalized = str(value).strip().upper()
    return LABEL_ALIASES_BY_ROLE[role].get(normalized, normalized)


def normalize_label_series(values: pd.Series, role: str) -> pd.Series:
    """Normalize source labels without mutating the supplied series."""

    validate_role(role)
    normalized = values.astype("string").str.strip().str.upper()
    return normalized.replace(dict(LABEL_ALIASES_BY_ROLE[role]))


def effective_target_labels(labels: Iterable[str], role: str) -> tuple[str, ...]:
    """Return governed target labels represented by raw source labels."""

    target_set = set(TARGET_LABELS)
    effective = {
        normalize_label_value(label, role)
        for label in labels
        if normalize_label_value(label, role) in target_set
    }
    return tuple(sorted(effective))
