"""Tests for A/B testing framework."""

from unittest.mock import MagicMock

from rag.ab_testing import ABTestFramework


class TestABTestFramework:
    def setup_method(self):
        self.ab = ABTestFramework(seed=42)
        self.retriever_a = MagicMock()
        self.retriever_b = MagicMock()
        self.retriever_a.retrieve.return_value = [{"text": "a", "score": 0.9}]
        self.retriever_b.retrieve.return_value = [{"text": "b", "score": 0.8}]

    def test_create_experiment(self):
        config = self.ab.create_experiment(
            "test_exp", {"A": self.retriever_a, "B": self.retriever_b}
        )
        assert config.name == "test_exp"
        assert config.active is True
        assert config.traffic_split == 0.5

    def test_create_experiment_custom_split(self):
        config = self.ab.create_experiment(
            "test_exp",
            {"A": self.retriever_a, "B": self.retriever_b},
            traffic_split=0.8,
        )
        assert config.traffic_split == 0.8

    def test_create_experiment_requires_two_variants(self):
        import pytest

        with pytest.raises(ValueError, match="Exactly 2 variants"):
            self.ab.create_experiment("bad", {"A": self.retriever_a})

    def test_create_experiment_validates_split(self):
        import pytest

        with pytest.raises(ValueError, match="traffic_split"):
            self.ab.create_experiment(
                "bad",
                {"A": self.retriever_a, "B": self.retriever_b},
                traffic_split=1.5,
            )

    def test_assign_returns_variant(self):
        self.ab.create_experiment("exp", {"A": self.retriever_a, "B": self.retriever_b})
        variant, obj = self.ab.assign("exp", "What is RAG?")
        assert variant in ("A", "B")
        assert obj in (self.retriever_a, self.retriever_b)

    def test_assign_deterministic(self):
        self.ab.create_experiment("exp", {"A": self.retriever_a, "B": self.retriever_b})
        v1, _ = self.ab.assign("exp", "same query")
        v2, _ = self.ab.assign("exp", "same query")
        assert v1 == v2

    def test_assign_different_queries_can_differ(self):
        self.ab.create_experiment(
            "exp",
            {"A": self.retriever_a, "B": self.retriever_b},
            traffic_split=0.5,
        )
        variants = set()
        for i in range(50):
            v, _ = self.ab.assign("exp", f"query_{i}")
            variants.add(v)
        # With 50 queries and 50/50 split, both variants should appear
        assert len(variants) == 2

    def test_assign_respects_split(self):
        # 100% to A
        self.ab.create_experiment(
            "all_a",
            {"A": self.retriever_a, "B": self.retriever_b},
            traffic_split=1.0,
        )
        for i in range(20):
            v, _ = self.ab.assign("all_a", f"q{i}")
            assert v == "A"

    def test_assign_unknown_experiment_raises(self):
        import pytest

        with pytest.raises(KeyError, match="not found"):
            self.ab.assign("nonexistent", "query")

    def test_stop_experiment(self):
        self.ab.create_experiment("exp", {"A": self.retriever_a, "B": self.retriever_b})
        assert self.ab.stop_experiment("exp") is True
        # Stopped experiment returns first variant
        v, _ = self.ab.assign("exp", "any query")
        assert v == "A"

    def test_stop_nonexistent(self):
        assert self.ab.stop_experiment("nope") is False

    def test_delete_experiment(self):
        import pytest

        self.ab.create_experiment("exp", {"A": self.retriever_a, "B": self.retriever_b})
        assert self.ab.delete_experiment("exp") is True
        with pytest.raises(KeyError):
            self.ab.assign("exp", "query")

    def test_delete_nonexistent(self):
        assert self.ab.delete_experiment("nope") is False

    def test_list_experiments(self):
        self.ab.create_experiment(
            "exp1", {"A": self.retriever_a, "B": self.retriever_b}
        )
        self.ab.create_experiment(
            "exp2", {"A": self.retriever_a, "B": self.retriever_b}
        )
        exps = self.ab.list_experiments()
        assert len(exps) == 2
        names = {e["name"] for e in exps}
        assert names == {"exp1", "exp2"}

    def test_record_and_get_results(self):
        self.ab.create_experiment("exp", {"A": self.retriever_a, "B": self.retriever_b})
        self.ab.record_outcome("exp", "A", {"latency": 0.2, "relevance": 0.9})
        self.ab.record_outcome("exp", "A", {"latency": 0.4, "relevance": 0.8})
        self.ab.record_outcome("exp", "B", {"latency": 0.3, "relevance": 0.7})

        results = self.ab.get_results("exp")
        assert results["total_outcomes"] == 3
        assert results["variants"]["A"]["count"] == 2
        assert abs(results["variants"]["A"]["averages"]["latency"] - 0.3) < 1e-9
        assert abs(results["variants"]["A"]["averages"]["relevance"] - 0.85) < 1e-9
        assert results["variants"]["B"]["count"] == 1

    def test_record_unknown_experiment_raises(self):
        import pytest

        with pytest.raises(KeyError):
            self.ab.record_outcome("nope", "A", {"latency": 0.1})

    def test_get_results_unknown_raises(self):
        import pytest

        with pytest.raises(KeyError):
            self.ab.get_results("nope")

    def test_get_results_empty(self):
        self.ab.create_experiment("exp", {"A": self.retriever_a, "B": self.retriever_b})
        results = self.ab.get_results("exp")
        assert results["total_outcomes"] == 0
        assert results["variants"] == {}

    def test_experiment_metadata_in_list(self):
        self.ab.create_experiment(
            "exp",
            {"A": self.retriever_a, "B": self.retriever_b},
            traffic_split=0.7,
        )
        self.ab.record_outcome("exp", "A", {"x": 1})
        exps = self.ab.list_experiments()
        assert exps[0]["traffic_split"] == 0.7
        assert exps[0]["total_outcomes"] == 1
        assert set(exps[0]["variants"]) == {"A", "B"}
