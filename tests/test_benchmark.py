import pytest
from eval.benchmark import keyword_recall, source_hit


class TestKeywordRecall:
    def test_all_keywords_found(self):
        assert keyword_recall("the cat sat on the mat", ["cat", "mat"]) == 1.0

    def test_no_keywords_found(self):
        assert keyword_recall("hello world", ["cat", "mat"]) == 0.0

    def test_partial_match(self):
        assert keyword_recall("the cat plays", ["cat", "mat"]) == 0.5

    def test_case_insensitive(self):
        assert keyword_recall("The CAT sat", ["cat"]) == 1.0

    def test_empty_keywords(self):
        assert keyword_recall("any answer", []) == 1.0


class TestSourceHit:
    def test_matching_source(self):
        citations = [{"title": "rag_intro.txt", "source": "data/raw/rag_intro.txt"}]
        assert source_hit(citations, "rag_intro.txt") is True

    def test_no_match(self):
        citations = [{"title": "other.txt", "source": "data/raw/other.txt"}]
        assert source_hit(citations, "rag_intro.txt") is False

    def test_none_expected(self):
        assert source_hit([], None) is True

    def test_empty_citations(self):
        assert source_hit([], "rag_intro.txt") is False
