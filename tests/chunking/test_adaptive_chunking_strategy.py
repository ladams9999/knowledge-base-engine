import pytest
from src.chunking.text_chunker import AdaptiveChunkingStrategy

def test_adaptive_chunking_not_implemented():
    with pytest.raises(NotImplementedError):
        AdaptiveChunkingStrategy(100, 10, 0)
