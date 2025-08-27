import pytest
from src.chunking.text_chunker import ParagraphChunkingStrategy

class DummyEmbeddingGenerator:
    def generate_embeddings_batch(self, texts):
        return [[1.0] * 10 for _ in texts]

def test_paragraph_chunking_basic():
    text = "Paragraph one.\n\nParagraph two is a bit longer.\n\nShort."
    chunker = ParagraphChunkingStrategy(chunk_size=30, min_chunk_size=10, overlap=0)
    chunks = chunker.chunk(text)
    assert isinstance(chunks, list)
    assert any("Paragraph one" in c for c in chunks)
    assert any("Paragraph two" in c for c in chunks)
    assert all(len(c) >= 10 for c in chunks)

def test_paragraph_chunking_large_paragraph():
    text = "This is a very long paragraph. " * 20
    chunker = ParagraphChunkingStrategy(chunk_size=50, min_chunk_size=20, overlap=0)
    chunks = chunker.chunk(text)
    assert all(len(c) >= 20 for c in chunks)
    assert any(len(c) <= 50 for c in chunks)

def test_paragraph_chunking_empty():
    chunker = ParagraphChunkingStrategy(chunk_size=30, min_chunk_size=10, overlap=0)
    assert chunker.chunk("") == []

def test_paragraph_chunking_min_chunk_size():
    text = "Short.\n\nAlso short."
    chunker = ParagraphChunkingStrategy(chunk_size=30, min_chunk_size=7, overlap=0)
    chunks = chunker.chunk(text)
    assert all(len(c) >= 7 for c in chunks)
