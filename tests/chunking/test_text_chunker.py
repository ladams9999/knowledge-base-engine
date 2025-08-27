import pytest
from src.chunking.text_chunker import TextChunker, ChunkingMethod

class DummyEmbeddingGenerator:
    def generate_embeddings_batch(self, texts):
        return [[1.0] * 10 for _ in texts]

def test_text_chunker_sentences():
    text = "This is a sentence. Here is another! And a third one? Short."
    chunker = TextChunker(chunk_size=30, min_chunk_size=10, overlap=0)
    chunks = chunker.chunk_text(text, method=ChunkingMethod.SENTENCES)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)
    assert any("sentence" in c for c in chunks)

def test_text_chunker_paragraphs():
    text = "Para one.\n\nPara two is longer."
    chunker = TextChunker(chunk_size=30, min_chunk_size=10, overlap=0)
    chunks = chunker.chunk_text(text, method=ChunkingMethod.PARAGRAPHS)
    assert isinstance(chunks, list)
    assert any("Para one" in c for c in chunks)

def test_text_chunker_semantic():
    text = "Sentence one. Sentence two. Sentence three."
    chunker = TextChunker(chunk_size=100, min_chunk_size=10, overlap=0, embedding_generator=DummyEmbeddingGenerator())
    chunks = chunker.chunk_text(text, method=ChunkingMethod.SEMANTIC)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)

def test_text_chunker_default_method():
    text = "This is a test."
    chunker = TextChunker(chunk_size=30, min_chunk_size=10, overlap=0)
    chunks = chunker.chunk_text(text)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)

def test_text_chunker_empty():
    chunker = TextChunker(chunk_size=30, min_chunk_size=10, overlap=0)
    assert chunker.chunk_text("") == []
