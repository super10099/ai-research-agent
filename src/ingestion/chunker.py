import uuid
from dataclasses import dataclass, field

import tiktoken


@dataclass
class Chunk:
    chunk_id: str
    text: str
    token_count: int
    source_url: str
    # None for parent chunks; parent chunk_id for child chunks.
    parent_id: str | None = field(default=None)


# cl100k_base is the BPE vocabulary used by Claude and GPT-4.
# Using it here means our token counts match the model's actual context usage.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _slide(
    tokens: list[int],
    window: int,
    overlap: float,
    min_tokens: int,
) -> list[list[int]]:
    """Sliding window tokenizer. Returns list of token slices."""
    stride = max(1, int(window * (1 - overlap)))
    slices = []
    for start in range(0, len(tokens), stride):
        chunk = tokens[start : start + window]
        if len(chunk) < min_tokens:
            break
        slices.append(chunk)
    return slices


def chunk_document(
    text: str,
    source_url: str,
    parent_size: int = 512,
    child_size: int = 256,
    overlap: float = 0.20,
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Hierarchical (small-to-big) chunking strategy.

    Returns (parent_chunks, child_chunks).

    Layout:
        Document tokens ──── parent window (512t, 20% overlap) ──▶ parent chunk
                                  └─── child window (256t, 20% overlap) ──▶ child chunk
                                            child.parent_id == parent.chunk_id

    At retrieval time we search child chunks (precise), then swap in their
    parents (more context) before sending to the model.  This is analogous to
    L1-cache-line-sized reads vs returning a full cache line to the CPU — the
    lookup key is small, the returned payload is larger.

    Why token-based and not character-based?
    Character counts vary wildly by language and punctuation. Token counts map
    directly to model context window consumption, so "256 tokens" means the
    same thing to the chunker and to Claude's context counter.
    """
    full_tokens = _ENCODING.encode(text)

    parent_chunks: list[Chunk] = []
    child_chunks: list[Chunk] = []

    for parent_tokens in _slide(full_tokens, parent_size, overlap, min_tokens=32):
        parent_id = str(uuid.uuid4())
        parent_text = _ENCODING.decode(parent_tokens)

        parent_chunks.append(Chunk(
            chunk_id=parent_id,
            text=parent_text,
            token_count=len(parent_tokens),
            source_url=source_url,
            parent_id=None,
        ))

        # Children are sliced from parent_tokens — they are always contained
        # within their parent, never spanning a parent boundary.
        for child_tokens in _slide(parent_tokens, child_size, overlap, min_tokens=16):
            child_chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=_ENCODING.decode(child_tokens),
                token_count=len(child_tokens),
                source_url=source_url,
                parent_id=parent_id,
            ))

    return parent_chunks, child_chunks
