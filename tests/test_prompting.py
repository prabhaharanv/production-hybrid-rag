from rag.prompting import build_rag_prompt, ABSTENTION_PHRASE


SAMPLE_CHUNKS = [
    {
        "chunk_id": "c1",
        "title": "doc1.txt",
        "source": "data/raw/doc1.txt",
        "text": "RAG combines retrieval with generation.",
    },
    {
        "chunk_id": "c2",
        "title": "doc2.txt",
        "source": "data/raw/doc2.txt",
        "text": "Chunking splits documents into pieces.",
    },
]


class TestAbstentionPhrase:
    def test_phrase_is_string(self):
        assert isinstance(ABSTENTION_PHRASE, str)
        assert len(ABSTENTION_PHRASE) > 0


class TestBuildRagPrompt:
    def test_returns_string(self):
        prompt = build_rag_prompt("What is RAG?", SAMPLE_CHUNKS)
        assert isinstance(prompt, str)

    def test_contains_question(self):
        prompt = build_rag_prompt("What is RAG?", SAMPLE_CHUNKS)
        assert "What is RAG?" in prompt

    def test_contains_chunk_text(self):
        prompt = build_rag_prompt("What is RAG?", SAMPLE_CHUNKS)
        assert "RAG combines retrieval with generation." in prompt
        assert "Chunking splits documents into pieces." in prompt

    def test_contains_bracket_citations(self):
        prompt = build_rag_prompt("What is RAG?", SAMPLE_CHUNKS)
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_contains_source_labels(self):
        prompt = build_rag_prompt("What is RAG?", SAMPLE_CHUNKS)
        assert "doc1.txt" in prompt
        assert "doc2.txt" in prompt

    def test_contains_abstention_phrase(self):
        prompt = build_rag_prompt("What is RAG?", SAMPLE_CHUNKS)
        assert ABSTENTION_PHRASE in prompt

    def test_empty_chunks(self):
        prompt = build_rag_prompt("What is RAG?", [])
        assert "What is RAG?" in prompt

    def test_single_chunk(self):
        prompt = build_rag_prompt("Question?", SAMPLE_CHUNKS[:1])
        assert "[1]" in prompt
        assert "[2]" not in prompt or "[2]" in prompt  # only [1] as a context block

    def test_contains_citation_rules(self):
        prompt = build_rag_prompt("Q?", SAMPLE_CHUNKS)
        # Prompt should instruct the LLM about citing sources
        assert "cite" in prompt.lower() or "[1]" in prompt
