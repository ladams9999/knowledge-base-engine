import logging
from typing import List

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles text chunking with configurable parameters and advanced methods."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
        min_chunk_size: int = 50,
        embedding_generator=None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.embedding_generator = embedding_generator
        self._advanced_chunker = None

    def _get_advanced_chunker(self):
        if self._advanced_chunker is None:
            try:
                from src.chunking import AdvancedTextChunker

                self._advanced_chunker = AdvancedTextChunker(
                    embedding_generator=self.embedding_generator,
                    chunk_size=self.chunk_size,
                    min_chunk_size=self.min_chunk_size,
                    overlap=self.overlap,
                )
            except ImportError:
                logger.warning(
                    "Advanced chunking module not available. Using basic chunking."
                )
                self._advanced_chunker = None
        return self._advanced_chunker

    def chunk_by_sentences(self, text: str) -> List[str]:
        sentences = []
        current_sentence = ""
        for char in text:
            current_sentence += char
            if char in ".!?" and len(current_sentence.strip()) > 10:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        return chunks

    def chunk_by_paragraphs(self, text: str) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) <= self.chunk_size:
                if len(paragraph) >= self.min_chunk_size:
                    chunks.append(paragraph)
            else:
                sentence_chunks = self.chunk_by_sentences(paragraph)
                chunks.extend(sentence_chunks)
        return chunks

    def chunk_text(self, text: str, method: str = "sentences") -> List[str]:
        if method in ["semantic", "gradient", "adaptive", "hierarchical"]:
            advanced_chunker = self._get_advanced_chunker()
            if advanced_chunker:
                try:
                    return advanced_chunker.chunk_text_adaptive(text, method)
                except Exception as e:
                    logger.warning(
                        f"Advanced chunking failed ({method}): {e}. Falling back to basic chunking."
                    )
                    method = "sentences"
            else:
                logger.warning(
                    "Advanced chunking not available. Falling back to basic method."
                )
                method = "sentences"
        if method == "paragraphs":
            return self.chunk_by_paragraphs(text)
        else:
            return self.chunk_by_sentences(text)
