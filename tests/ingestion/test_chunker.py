"""Tests for src/ingestion/chunker.py — hierarchical (small-to-big) chunking."""

from src.ingestion.chunker import _ENCODING, _slide, chunk_document


class TestSlide:
    def test_basic_sliding_window(self):
        tokens = list(range(1000))
        slices = _slide(tokens, window=100, overlap=0.2, min_tokens=10)

        # stride = window * (1 - overlap) = 100 * 0.8 = 80
        assert slices[0] == list(range(0, 100))
        assert slices[1] == list(range(80, 180))

    def test_drops_tail_shorter_than_min_tokens(self):
        tokens = list(range(105))  # last window at start=80 has only 25 tokens
        slices = _slide(tokens, window=100, overlap=0.2, min_tokens=30)

        # start=0 -> 100 tokens (kept), start=80 -> 25 tokens (< min_tokens=30, dropped)
        assert len(slices) == 1

    def test_keeps_short_tail_when_above_min_tokens(self):
        tokens = list(range(105))
        slices = _slide(tokens, window=100, overlap=0.2, min_tokens=20)

        assert len(slices) == 2
        assert slices[1] == list(range(80, 105))

    def test_empty_input(self):
        assert _slide([], window=100, overlap=0.2, min_tokens=10) == []


class TestChunkDocument:
    def _long_text(self, paragraphs: int = 40) -> str:
        # Each paragraph is long enough that ~40 of them comfortably exceed
        # parent_size=512 tokens, giving us multiple parents to test against.
        paragraph = (
            "Retrieval-augmented generation combines a parametric language model "
            "with a non-parametric retrieval mechanism over an external corpus. "
        )
        return "\n\n".join(f"{paragraph} Section {i}." for i in range(paragraphs))

    def test_produces_multiple_parents_for_long_document(self):
        parents, children = chunk_document(self._long_text(), source_url="http://x.test")

        assert len(parents) >= 2
        assert len(children) > 0

    def test_every_child_parent_id_matches_a_real_parent(self):
        parents, children = chunk_document(self._long_text(), source_url="http://x.test")
        parent_ids = {p.chunk_id for p in parents}

        assert all(c.parent_id in parent_ids for c in children)

    def test_every_parent_has_at_least_one_child(self):
        parents, children = chunk_document(self._long_text(), source_url="http://x.test")
        parent_ids_with_children = {c.parent_id for c in children}

        assert all(p.chunk_id in parent_ids_with_children for p in parents)

    def test_chunk_ids_are_unique(self):
        parents, children = chunk_document(self._long_text(), source_url="http://x.test")
        all_ids = [p.chunk_id for p in parents] + [c.chunk_id for c in children]

        assert len(all_ids) == len(set(all_ids))

    def test_token_counts_within_bounds(self):
        parents, children = chunk_document(
            self._long_text(), source_url="http://x.test", parent_size=512, child_size=256,
        )

        assert all(p.token_count <= 512 for p in parents)
        assert all(c.token_count <= 256 for c in children)

    def test_source_url_propagated(self):
        parents, children = chunk_document(self._long_text(), source_url="http://x.test/paper")

        assert all(p.source_url == "http://x.test/paper" for p in parents)
        assert all(c.source_url == "http://x.test/paper" for c in children)

    def test_children_never_cross_parent_boundary(self):
        """A child's tokens must be a contiguous sub-slice of its parent's tokens —
        never spanning into the next parent window. We verify this by re-encoding
        each child's text and confirming it is a contiguous subsequence of its
        parent's re-encoded tokens."""
        parents, children = chunk_document(self._long_text(), source_url="http://x.test")
        parent_tokens_by_id = {p.chunk_id: _ENCODING.encode(p.text) for p in parents}

        for c in children:
            parent_tokens = parent_tokens_by_id[c.parent_id]
            child_tokens = _ENCODING.encode(c.text)
            joined_parent = " ".join(map(str, parent_tokens))
            joined_child = " ".join(map(str, child_tokens))
            assert joined_child in joined_parent

    def test_short_text_produces_no_chunks(self):
        # Well under min_tokens=32 for the parent window.
        parents, children = chunk_document("Too short.", source_url="http://x.test")

        assert parents == []
        assert children == []
