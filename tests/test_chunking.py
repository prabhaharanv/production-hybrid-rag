import pytest
from rag.chunking import (
    chunk_text,
    chunk_documents,
    get_chunking_strategy,
    WordChunking,
    SentenceChunking,
    RecursiveChunking,
    TokenChunking,
    STRATEGIES,
)


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []

    def test_short_text_single_chunk(self):
        result = chunk_text("hello world", chunk_size=10)
        assert len(result) == 1
        assert result[0] == "hello world"

    def test_exact_chunk_size(self):
        text = " ".join(["word"] * 10)
        result = chunk_text(text, chunk_size=10, overlap=0)
        assert len(result) == 1

    def test_overlap_produces_more_chunks(self):
        text = " ".join(["word"] * 20)
        no_overlap = chunk_text(text, chunk_size=10, overlap=0)
        with_overlap = chunk_text(text, chunk_size=10, overlap=5)
        assert len(with_overlap) > len(no_overlap)

    def test_chunks_contain_all_words(self):
        words = [f"w{i}" for i in range(15)]
        text = " ".join(words)
        result = chunk_text(text, chunk_size=10, overlap=3)
        all_text = " ".join(result)
        for w in words:
            assert w in all_text

    def test_strategy_parameter(self):
        text = " ".join(["word"] * 20)
        result = chunk_text(text, chunk_size=10, overlap=2, strategy="word")
        assert len(result) >= 2


class TestChunkDocuments:
    def test_single_document(self):
        docs = [
            {
                "doc_id": "d1",
                "title": "test",
                "source": "test.txt",
                "text": "hello world",
            }
        ]
        result = chunk_documents(docs, chunk_size=100)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "d1_chunk_0"
        assert result[0]["doc_id"] == "d1"

    def test_chunk_ids_are_unique(self):
        docs = [
            {
                "doc_id": "d1",
                "title": "a",
                "source": "a.txt",
                "text": " ".join(["word"] * 30),
            },
            {
                "doc_id": "d2",
                "title": "b",
                "source": "b.txt",
                "text": " ".join(["word"] * 30),
            },
        ]
        result = chunk_documents(docs, chunk_size=10, overlap=2)
        ids = [c["chunk_id"] for c in result]
        assert len(ids) == len(set(ids))

    def test_metadata_has_chunk_index(self):
        docs = [
            {
                "doc_id": "d1",
                "title": "t",
                "source": "s",
                "text": " ".join(["word"] * 30),
            }
        ]
        result = chunk_documents(docs, chunk_size=10, overlap=2)
        for i, chunk in enumerate(result):
            assert chunk["metadata"]["chunk_index"] == i

    def test_strategy_parameter_in_chunk_documents(self):
        docs = [
            {
                "doc_id": "d1",
                "title": "t",
                "source": "s",
                "text": " ".join(["word"] * 20),
            }
        ]
        result = chunk_documents(docs, chunk_size=10, overlap=2, strategy="sentence")
        assert len(result) >= 1


# ---- Strategy-specific tests ----


class TestWordChunking:
    def test_basic_split(self):
        s = WordChunking()
        text = " ".join(["word"] * 20)
        chunks = s.chunk(text, chunk_size=10, overlap=0)
        assert len(chunks) == 2
        assert all(len(c.split()) <= 10 for c in chunks)

    def test_overlap(self):
        s = WordChunking()
        text = " ".join([f"w{i}" for i in range(20)])
        chunks = s.chunk(text, chunk_size=10, overlap=3)
        # Check overlap: last 3 words of chunk 0 should appear at start of chunk 1
        words_0 = chunks[0].split()
        words_1 = chunks[1].split()
        assert words_0[-3:] == words_1[:3]

    def test_empty_text(self):
        s = WordChunking()
        assert s.chunk("", chunk_size=10, overlap=2) == []


class TestSentenceChunking:
    def test_respects_sentence_boundaries(self):
        s = SentenceChunking()
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = s.chunk(text, chunk_size=5, overlap=0)
        # Each sentence has 3 words, chunk_size=5 should group ~1-2 sentences per chunk
        assert len(chunks) >= 2
        # No sentence should be split mid-sentence
        for chunk in chunks:
            assert chunk.endswith(".") or chunk == chunks[-1]

    def test_single_sentence(self):
        s = SentenceChunking()
        text = "Just one sentence."
        chunks = s.chunk(text, chunk_size=10, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == "Just one sentence."

    def test_overlap_keeps_sentences(self):
        s = SentenceChunking()
        text = "A short one. Another short one. Third one here. Fourth one here. Fifth one here."
        chunks = s.chunk(text, chunk_size=4, overlap=3)
        assert len(chunks) >= 2

    def test_empty_text(self):
        s = SentenceChunking()
        result = s.chunk("", chunk_size=10, overlap=2)
        # Empty or whitespace-only input returns single empty-ish chunk
        assert all(c.strip() == "" for c in result) or result == []


class TestRecursiveChunking:
    def test_splits_on_paragraphs_first(self):
        s = RecursiveChunking()
        text = "First paragraph content here.\n\nSecond paragraph content here."
        chunks = s.chunk(text, chunk_size=5, overlap=0)
        assert len(chunks) == 2

    def test_falls_back_to_sentences(self):
        s = RecursiveChunking()
        text = (
            "Long paragraph. With many sentences. That go on. And on. And on further."
        )
        chunks = s.chunk(text, chunk_size=4, overlap=0)
        assert len(chunks) >= 2

    def test_short_text_single_chunk(self):
        s = RecursiveChunking()
        text = "Short text"
        chunks = s.chunk(text, chunk_size=10, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_empty_text(self):
        s = RecursiveChunking()
        assert s.chunk("", chunk_size=10, overlap=0) == []


class TestTokenChunking:
    def test_basic_split(self):
        s = TokenChunking()
        # 50 words should produce multiple token chunks at chunk_size=20
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = s.chunk(text, chunk_size=20, overlap=0)
        assert len(chunks) >= 2

    def test_overlap(self):
        s = TokenChunking()
        text = " ".join([f"word{i}" for i in range(50)])
        no_overlap = s.chunk(text, chunk_size=20, overlap=0)
        with_overlap = s.chunk(text, chunk_size=20, overlap=5)
        assert len(with_overlap) > len(no_overlap)

    def test_reconstructs_text(self):
        s = TokenChunking()
        text = "The quick brown fox jumps over the lazy dog."
        chunks = s.chunk(text, chunk_size=5, overlap=0)
        reconstructed = "".join(chunks)
        assert reconstructed == text

    def test_empty_text(self):
        s = TokenChunking()
        assert s.chunk("", chunk_size=10, overlap=0) == []


class TestStrategyRegistry:
    def test_all_strategies_registered(self):
        assert "word" in STRATEGIES
        assert "sentence" in STRATEGIES
        assert "recursive" in STRATEGIES
        assert "token" in STRATEGIES

    def test_get_chunking_strategy_default(self):
        s = get_chunking_strategy("word")
        assert isinstance(s, WordChunking)

    def test_get_chunking_strategy_all(self):
        assert isinstance(get_chunking_strategy("word"), WordChunking)
        assert isinstance(get_chunking_strategy("sentence"), SentenceChunking)
        assert isinstance(get_chunking_strategy("recursive"), RecursiveChunking)
        assert isinstance(get_chunking_strategy("token"), TokenChunking)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunking_strategy("nonexistent")
