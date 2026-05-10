from rag.pipeline import RAGPipeline
from rag.prompting import ABSTENTION_PHRASE


# ---- Stubs ----


class FakeRetriever:
    def __init__(self, chunks=None):
        self._chunks = (
            chunks
            if chunks is not None
            else [
                {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "title": "doc.txt",
                    "source": "data/raw/doc.txt",
                    "text": "RAG combines retrieval and generation.",
                    "score": 0.9,
                    "metadata": {},
                },
                {
                    "chunk_id": "c2",
                    "doc_id": "d2",
                    "title": "other.txt",
                    "source": "data/raw/other.txt",
                    "text": "Chunking is important for RAG.",
                    "score": 0.7,
                    "metadata": {},
                },
            ]
        )

    def retrieve(self, query, top_k=5):
        return self._chunks[:top_k]


class FakeGenerator:
    def __init__(self, answer="RAG is great [1]."):
        self._answer = answer

    def generate(self, prompt):
        return self._answer


class FakeReranker:
    def rerank(self, query, chunks, top_k=5):
        for i, c in enumerate(chunks):
            c["rerank_score"] = 1.0 - i * 0.1
        return chunks[:top_k]


class FakeQueryRewriter:
    def rewrite(self, query):
        return f"rewritten: {query}"


# ---- Tests ----


class TestRAGPipeline:
    def test_basic_ask(self):
        pipe = RAGPipeline(retriever=FakeRetriever(), generator=FakeGenerator())
        result = pipe.ask("What is RAG?")
        assert result["question"] == "What is RAG?"
        assert result["answer"] == "RAG is great [1]."
        assert result["abstained"] is False

    def test_returns_all_required_keys(self):
        pipe = RAGPipeline(retriever=FakeRetriever(), generator=FakeGenerator())
        result = pipe.ask("What is RAG?")
        required_keys = {
            "question",
            "rewritten_query",
            "answer",
            "abstained",
            "citations",
            "retrieved_chunks",
        }
        assert required_keys.issubset(result.keys())

    def test_citation_extraction(self):
        pipe = RAGPipeline(
            retriever=FakeRetriever(),
            generator=FakeGenerator(answer="Answer [1][2]."),
        )
        result = pipe.ask("Q?")
        refs = [c["reference"] for c in result["citations"]]
        assert 1 in refs
        assert 2 in refs

    def test_citation_out_of_range_ignored(self):
        pipe = RAGPipeline(
            retriever=FakeRetriever(),
            generator=FakeGenerator(answer="Answer [99]."),
        )
        result = pipe.ask("Q?")
        assert len(result["citations"]) == 0

    def test_abstention(self):
        pipe = RAGPipeline(
            retriever=FakeRetriever(),
            generator=FakeGenerator(answer=ABSTENTION_PHRASE),
        )
        result = pipe.ask("What is quantum physics?")
        assert result["abstained"] is True
        assert ABSTENTION_PHRASE not in result["answer"]

    def test_query_rewriting(self):
        pipe = RAGPipeline(
            retriever=FakeRetriever(),
            generator=FakeGenerator(),
            query_rewriter=FakeQueryRewriter(),
        )
        result = pipe.ask("RAG?")
        assert result["rewritten_query"] == "rewritten: RAG?"

    def test_no_query_rewriter(self):
        pipe = RAGPipeline(retriever=FakeRetriever(), generator=FakeGenerator())
        result = pipe.ask("RAG?")
        assert result["rewritten_query"] == "RAG?"

    def test_with_reranker(self):
        pipe = RAGPipeline(
            retriever=FakeRetriever(),
            generator=FakeGenerator(),
            reranker=FakeReranker(),
        )
        result = pipe.ask("What is RAG?", top_k=2)
        assert len(result["retrieved_chunks"]) <= 2

    def test_reranker_fetches_extra_candidates(self):
        """When reranker is present, pipeline should fetch top_k * 3 candidates."""
        chunks = [
            {
                "chunk_id": f"c{i}",
                "doc_id": f"d{i}",
                "title": f"doc{i}.txt",
                "source": f"data/raw/doc{i}.txt",
                "text": f"Text {i}",
                "score": 0.9 - i * 0.05,
                "metadata": {},
            }
            for i in range(15)
        ]

        class TrackingRetriever:
            def __init__(self):
                self.last_top_k = None

            def retrieve(self, query, top_k=5):
                self.last_top_k = top_k
                return chunks[:top_k]

        tracker = TrackingRetriever()
        pipe = RAGPipeline(
            retriever=tracker,
            generator=FakeGenerator(),
            reranker=FakeReranker(),
        )
        pipe.ask("Q?", top_k=3)
        assert tracker.last_top_k == 9  # 3 * 3

    def test_empty_retrieval(self):
        pipe = RAGPipeline(
            retriever=FakeRetriever(chunks=[]),
            generator=FakeGenerator(answer=ABSTENTION_PHRASE),
        )
        result = pipe.ask("Q?")
        assert result["abstained"] is True
        assert result["retrieved_chunks"] == []
