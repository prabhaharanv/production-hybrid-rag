"""Tests for conversation memory."""

import time
from rag.memory import ConversationMemory, Turn, build_history_context


class TestConversationMemory:
    def setup_method(self):
        self.memory = ConversationMemory(max_turns=3, ttl=60)

    def test_add_and_get_turn(self):
        self.memory.add_turn("c1", "What is RAG?", "RAG is ...")
        turns = self.memory.get_history("c1")
        assert len(turns) == 1
        assert turns[0].question == "What is RAG?"
        assert turns[0].answer == "RAG is ..."

    def test_sliding_window(self):
        for i in range(5):
            self.memory.add_turn("c1", f"q{i}", f"a{i}")
        turns = self.memory.get_history("c1")
        assert len(turns) == 3
        assert turns[0].question == "q2"
        assert turns[-1].question == "q4"

    def test_separate_conversations(self):
        self.memory.add_turn("c1", "q1", "a1")
        self.memory.add_turn("c2", "q2", "a2")
        assert len(self.memory.get_history("c1")) == 1
        assert len(self.memory.get_history("c2")) == 1
        assert self.memory.get_history("c1")[0].question == "q1"

    def test_clear_single(self):
        self.memory.add_turn("c1", "q", "a")
        self.memory.add_turn("c2", "q", "a")
        self.memory.clear("c1")
        assert len(self.memory.get_history("c1")) == 0
        assert len(self.memory.get_history("c2")) == 1

    def test_clear_all(self):
        self.memory.add_turn("c1", "q", "a")
        self.memory.add_turn("c2", "q", "a")
        self.memory.clear_all()
        assert len(self.memory.get_history("c1")) == 0
        assert len(self.memory.get_history("c2")) == 0

    def test_list_conversations(self):
        self.memory.add_turn("c1", "q1", "a1")
        self.memory.add_turn("c2", "q2", "a2")
        convos = self.memory.list_conversations()
        assert len(convos) == 2
        names = {c["conversation_id"] for c in convos}
        assert names == {"c1", "c2"}
        for c in convos:
            assert c["turn_count"] == 1
            assert c["last_activity"] > 0

    def test_evict_stale(self):
        self.memory.add_turn("c1", "q1", "a1")
        # Manually backdate the timestamp
        self.memory._conversations["c1"][0].timestamp = time.time() - 120
        evicted = self.memory.evict_stale()
        assert evicted == 1
        assert len(self.memory.get_history("c1")) == 0

    def test_evict_stale_keeps_recent(self):
        self.memory.add_turn("c1", "q1", "a1")
        evicted = self.memory.evict_stale()
        assert evicted == 0
        assert len(self.memory.get_history("c1")) == 1

    def test_empty_history(self):
        assert self.memory.get_history("nonexistent") == []

    def test_no_ttl_skips_eviction(self):
        no_ttl = ConversationMemory(max_turns=3, ttl=0)
        no_ttl.add_turn("c1", "q", "a")
        no_ttl._conversations["c1"][0].timestamp = 0
        assert no_ttl.evict_stale() == 0


class TestBuildHistoryContext:
    def test_empty_turns(self):
        assert build_history_context([]) == ""

    def test_formats_turns(self):
        turns = [
            Turn(question="What is RAG?", answer="RAG is ..."),
            Turn(question="How does it work?", answer="It retrieves ..."),
        ]
        result = build_history_context(turns)
        assert "Previous conversation:" in result
        assert "User: What is RAG?" in result
        assert "Assistant: RAG is ..." in result
        assert "User: How does it work?" in result
        assert "Assistant: It retrieves ..." in result

    def test_single_turn(self):
        turns = [Turn(question="Hello", answer="Hi")]
        result = build_history_context(turns)
        assert "User: Hello" in result
        assert "Assistant: Hi" in result
