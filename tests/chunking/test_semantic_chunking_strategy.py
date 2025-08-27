import pytest
from src.chunking.text_chunker import SemanticChunkingStrategy

class DummyEmbeddingGenerator:
    def generate_embeddings_batch(self, texts):
        # Return orthogonal vectors for each sentence to force boundaries
        return [[float(i == j) for i in range(len(texts))] for j in range(len(texts))]

def test_semantic_chunking_basic():
    text = "Sentence one is here. Sentence two is different. Sentence three is unique."
    chunker = SemanticChunkingStrategy(chunk_size=100, min_chunk_size=10, overlap=0, embedding_generator=DummyEmbeddingGenerator())
    chunks = chunker.chunk(text)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) >= 10 for c in chunks)
    assert sum("Sentence one" in c for c in chunks) == 1

def test_semantic_chunking_no_embeddings():
    text = "Sentence one. Sentence two. Sentence three."
    chunker = SemanticChunkingStrategy(chunk_size=100, min_chunk_size=10, overlap=0, embedding_generator=None)
    chunks = chunker.chunk(text)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)

def test_semantic_chunking_insufficient_embeddings():
    class BadEmbeddingGen:
        def generate_embeddings_batch(self, texts):
            return [None for _ in texts]
    text = "Sentence one. Sentence two. Sentence three."
    chunker = SemanticChunkingStrategy(chunk_size=100, min_chunk_size=10, overlap=0, embedding_generator=BadEmbeddingGen())
    chunks = chunker.chunk(text)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)

def test_semantic_chunking_empty():
    chunker = SemanticChunkingStrategy(chunk_size=100, min_chunk_size=10, overlap=0, embedding_generator=DummyEmbeddingGenerator())
    assert chunker.chunk("") == []
