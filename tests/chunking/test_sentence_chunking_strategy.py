import pytest
from src.chunking.text_chunker import SentenceChunkingStrategy

class DummyEmbeddingGenerator:
    def generate_embeddings_batch(self, texts):
        return [[1.0] * 10 for _ in texts]

def test_sentence_chunking_basic():
    text = "This is a sentence. Here is another! And a third one? Short."
    chunker = SentenceChunkingStrategy(chunk_size=30, min_chunk_size=10, overlap=0)
    chunks = chunker.chunk(text)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)
    assert any("sentence" in c for c in chunks)
    assert any("another" in c for c in chunks)
    assert all(len(c) >= 10 for c in chunks)

def test_sentence_chunking_handles_short_sentences():
    text = "Hi. Ok. No."
    chunker = SentenceChunkingStrategy(chunk_size=20, min_chunk_size=2, overlap=0)
    chunks = chunker.chunk(text)
    assert all(len(c) >= 2 for c in chunks)

def test_sentence_chunking_empty():
    chunker = SentenceChunkingStrategy(chunk_size=20, min_chunk_size=2, overlap=0)
    assert chunker.chunk("") == []

def test_sentence_chunking_min_chunk_size():
    text = "A very long sentence that should be its own chunk."
    chunker = SentenceChunkingStrategy(chunk_size=100, min_chunk_size=40, overlap=0)
    chunks = chunker.chunk(text)
    assert all(len(c) >= 40 for c in chunks)

def test_sentence_chunking_exact_chunk_size():
    text = "This is a test sentence. " * 5
    chunker = SentenceChunkingStrategy(chunk_size=len(text)//2, min_chunk_size=10, overlap=0)
    chunks = chunker.chunk(text)
    assert all(len(c) >= 10 for c in chunks)
