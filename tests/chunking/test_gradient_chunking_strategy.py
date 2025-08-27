import pytest
from src.chunking.text_chunker import GradientChunkingStrategy

def test_gradient_chunking_not_implemented():
    with pytest.raises(NotImplementedError):
        GradientChunkingStrategy(100, 10, 0)
