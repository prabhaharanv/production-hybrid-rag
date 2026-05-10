"""A/B testing framework for retrieval strategies.

Randomly assigns incoming queries to experiment variants (A or B),
logs the assignment, and exposes metrics to compare strategies.
Assignments are deterministic per (experiment_id, query) for reproducibility.

Usage:
    ab = ABTestFramework()
    ab.create_experiment("hybrid_vs_dense", {"A": hybrid_retriever, "B": dense_retriever})
    variant, retriever = ab.assign("hybrid_vs_dense", query)
    results = retriever.retrieve(query, top_k=5)
    ab.record_outcome("hybrid_vs_dense", variant, {"latency": 0.3, "relevance": 0.85})
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field

from app.observability.logging import get_logger

log = get_logger("rag.ab_testing")


@dataclass
class ExperimentConfig:
    """Configuration for a single A/B experiment."""

    name: str
    variants: dict[str, object]  # variant_name → retriever (or any strategy object)
    traffic_split: float = 0.5  # fraction routed to variant "A"
    active: bool = True


@dataclass
class OutcomeRecord:
    """A single recorded outcome for an experiment variant."""

    variant: str
    metrics: dict[str, float]
    timestamp: float = field(default_factory=time.time)


class ABTestFramework:
    """Manages A/B experiments for retrieval strategies.

    Parameters
    ----------
    seed : int
        Hash seed for deterministic assignment.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._experiments: dict[str, ExperimentConfig] = {}
        self._outcomes: dict[str, list[OutcomeRecord]] = {}
        self._lock = threading.Lock()

    # --- Experiment lifecycle ---

    def create_experiment(
        self,
        name: str,
        variants: dict[str, object],
        traffic_split: float = 0.5,
    ) -> ExperimentConfig:
        """Register a new A/B experiment.

        Parameters
        ----------
        name : str
            Unique experiment name.
        variants : dict
            Exactly two entries, e.g. ``{"A": retriever_a, "B": retriever_b}``.
        traffic_split : float
            Fraction of traffic routed to variant "A" (0.0–1.0).
        """
        if len(variants) != 2:
            raise ValueError("Exactly 2 variants required for A/B test")
        if not 0.0 <= traffic_split <= 1.0:
            raise ValueError("traffic_split must be between 0.0 and 1.0")

        config = ExperimentConfig(
            name=name,
            variants=variants,
            traffic_split=traffic_split,
        )
        with self._lock:
            self._experiments[name] = config
            self._outcomes.setdefault(name, [])
        log.info("experiment_created", experiment=name, split=traffic_split)
        return config

    def stop_experiment(self, name: str) -> bool:
        """Deactivate an experiment (keeps data). Returns False if not found."""
        with self._lock:
            exp = self._experiments.get(name)
            if exp is None:
                return False
            exp.active = False
        log.info("experiment_stopped", experiment=name)
        return True

    def delete_experiment(self, name: str) -> bool:
        """Remove an experiment and its data."""
        with self._lock:
            removed = self._experiments.pop(name, None)
            self._outcomes.pop(name, None)
        if removed:
            log.info("experiment_deleted", experiment=name)
        return removed is not None

    def list_experiments(self) -> list[dict]:
        """Return metadata for all experiments."""
        with self._lock:
            return [
                {
                    "name": exp.name,
                    "active": exp.active,
                    "traffic_split": exp.traffic_split,
                    "variants": list(exp.variants.keys()),
                    "total_outcomes": len(self._outcomes.get(exp.name, [])),
                }
                for exp in self._experiments.values()
            ]

    # --- Assignment ---

    def _hash_to_bucket(self, experiment: str, query: str) -> float:
        """Deterministic hash of (experiment, query) → [0.0, 1.0)."""
        raw = f"{self.seed}:{experiment}:{query}"
        h = hashlib.sha256(raw.encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def assign(self, experiment_name: str, query: str) -> tuple[str, object]:
        """Assign a query to a variant. Returns (variant_name, strategy_object).

        Raises KeyError if the experiment doesn't exist.
        """
        with self._lock:
            exp = self._experiments.get(experiment_name)
            if exp is None:
                raise KeyError(f"Experiment '{experiment_name}' not found")
            if not exp.active:
                # Inactive experiment → always return first variant
                first_key = next(iter(exp.variants))
                return first_key, exp.variants[first_key]

        bucket = self._hash_to_bucket(experiment_name, query)
        variant_keys = sorted(exp.variants.keys())
        variant_name = (
            variant_keys[0] if bucket < exp.traffic_split else variant_keys[1]
        )

        log.debug(
            "ab_assignment",
            experiment=experiment_name,
            variant=variant_name,
            bucket=round(bucket, 4),
        )
        return variant_name, exp.variants[variant_name]

    # --- Outcome tracking ---

    def record_outcome(
        self,
        experiment_name: str,
        variant: str,
        metrics: dict[str, float],
    ) -> None:
        """Record an outcome for a variant."""
        with self._lock:
            outcomes = self._outcomes.get(experiment_name)
            if outcomes is None:
                raise KeyError(f"Experiment '{experiment_name}' not found")
            outcomes.append(OutcomeRecord(variant=variant, metrics=metrics))

    def get_results(self, experiment_name: str) -> dict:
        """Aggregate results for an experiment.

        Returns per-variant averages of all recorded metrics.
        """
        with self._lock:
            outcomes = self._outcomes.get(experiment_name)
            if outcomes is None:
                raise KeyError(f"Experiment '{experiment_name}' not found")
            outcomes = list(outcomes)  # snapshot

        # Group by variant
        by_variant: dict[str, list[dict[str, float]]] = {}
        for o in outcomes:
            by_variant.setdefault(o.variant, []).append(o.metrics)

        summary: dict[str, dict] = {}
        for variant, records in by_variant.items():
            agg: dict[str, float] = {}
            for rec in records:
                for k, v in rec.items():
                    agg[k] = agg.get(k, 0.0) + v
            count = len(records)
            summary[variant] = {
                "count": count,
                "averages": {k: v / count for k, v in agg.items()},
            }

        return {
            "experiment": experiment_name,
            "total_outcomes": len(outcomes),
            "variants": summary,
        }
