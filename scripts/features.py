"""GDELT cluster → feature vector for classical ML models.

Each cluster (one week of NK military events) is reduced to a fixed-length
numeric feature vector suitable for XGBoost or any sklearn estimator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# CAMEO root codes present in the dataset (order defines feature columns)
CAMEO_CODES = ["14", "15", "16", "17", "18", "19", "20"]


def cluster_to_features(cluster: dict) -> np.ndarray:
    """Convert a GDELT cluster dict to a 1-D feature array.

    Features (13 total):
      0   avg_goldstein       mean Goldstein scale for the week
      1   avg_tone            mean AvgTone
      2   total_events        total event count
      3   total_mentions      total NumMentions
      4   total_sources       total NumSources
      5   log_events          log1p(total_events)
      6   log_mentions        log1p(total_mentions)
      7-13  cameo_frac_*      fraction of events per CAMEO root code
    """
    stats = cluster.get("stats", {})

    total_events = stats.get("total_events", 0) or 0
    total_mentions = stats.get("total_mentions", 0) or 0
    total_sources = stats.get("total_sources", 0) or 0
    avg_goldstein = stats.get("avg_goldstein", 0.0) or 0.0
    avg_tone = stats.get("avg_tone", 0.0) or 0.0

    breakdown = cluster.get("event_breakdown", {})
    cameo_fracs = []
    for code in CAMEO_CODES:
        count = breakdown.get(code, 0) or 0
        cameo_fracs.append(count / total_events if total_events > 0 else 0.0)

    return np.array(
        [
            avg_goldstein,
            avg_tone,
            total_events,
            total_mentions,
            total_sources,
            np.log1p(total_events),
            np.log1p(total_mentions),
            *cameo_fracs,
        ],
        dtype=np.float32,
    )


FEATURE_NAMES = [
    "avg_goldstein",
    "avg_tone",
    "total_events",
    "total_mentions",
    "total_sources",
    "log_events",
    "log_mentions",
    *(f"cameo_frac_{c}" for c in CAMEO_CODES),
]


def load_feature_matrix(
    clusters_path: str | Path,
    labeled_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and align clusters + labels into (X, y, week_ids).

    Returns:
        X         shape (N, 13) float32 feature matrix
        y         shape (N,)    int escalation_level labels (1-5)
        week_ids  list of week start strings for traceability
    """
    clusters: dict[str, dict] = {}
    for line in Path(clusters_path).read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            clusters[c["week_start"]] = c

    X_rows, y_rows, week_ids = [], [], []
    for line in Path(labeled_path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        week = item["week_start"]
        label = item.get("assessment", {}).get("escalation_level")
        if label is None or week not in clusters:
            continue
        X_rows.append(cluster_to_features(clusters[week]))
        y_rows.append(int(label))
        week_ids.append(week)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), week_ids
