import pytest
from src.chunking.text_chunker import ChunkingStrategyFactory, ChunkingMethod, SentenceChunkingStrategy, ParagraphChunkingStrategy, SemanticChunkingStrategy

def test_factory_returns_correct_strategy():
    s = ChunkingStrategyFactory.get_strategy(ChunkingMethod.SENTENCES, 100, 10, 0)
    assert isinstance(s, SentenceChunkingStrategy)
    p = ChunkingStrategyFactory.get_strategy(ChunkingMethod.PARAGRAPHS, 100, 10, 0)
    assert isinstance(p, ParagraphChunkingStrategy)
    sem = ChunkingStrategyFactory.get_strategy(ChunkingMethod.SEMANTIC, 100, 10, 0)
    assert isinstance(sem, SemanticChunkingStrategy)
    with pytest.raises(NotImplementedError):
        ChunkingStrategyFactory.get_strategy(ChunkingMethod.GRADIENT, 100, 10, 0)
    with pytest.raises(NotImplementedError):
        ChunkingStrategyFactory.get_strategy(ChunkingMethod.ADAPTIVE, 100, 10, 0)

def test_factory_default_to_sentence():
    s = ChunkingStrategyFactory.get_strategy("unknown", 100, 10, 0)
    assert isinstance(s, SentenceChunkingStrategy)
