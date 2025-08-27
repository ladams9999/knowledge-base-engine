import logging
from typing import List, Dict, Type
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ChunkingMethod:
    SENTENCES = "sentences"
    PARAGRAPHS = "paragraphs"
    SEMANTIC = "semantic"
    GRADIENT = "gradient"
    ADAPTIVE = "adaptive"


class ChunkingStrategy(ABC):
    def __init__(
        self,
        chunk_size: int,
        min_chunk_size: int,
        overlap: int,
        embedding_generator=None,
    ):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.embedding_generator = embedding_generator

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass


class SentenceChunkingStrategy(ChunkingStrategy):
    def chunk(self, text: str) -> List[str]:
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


class ParagraphChunkingStrategy(ChunkingStrategy):
    def chunk(self, text: str) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks = []
        # Use sentence chunking for large paragraphs
        sentence_chunker = SentenceChunkingStrategy(
            self.chunk_size, self.min_chunk_size, self.overlap
        )
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) <= self.chunk_size:
                if len(paragraph) >= self.min_chunk_size:
                    chunks.append(paragraph)
            else:
                sentence_chunks = sentence_chunker.chunk(paragraph)
                chunks.extend(sentence_chunks)
        return chunks


class SemanticChunkingStrategy(ChunkingStrategy):
    def chunk(self, text: str) -> List[str]:
        def extract_sentences(text: str):
            import re

            sentences = []
            current_sentence = ""
            for char in text:
                current_sentence += char
                if char in ".!?" and len(current_sentence.strip()) > 10:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            # Add start/end positions for compatibility
            pos = 0
            result = []
            for s in sentences:
                start = text.find(s, pos)
                end = start + len(s)
                result.append((s, start, end))
                pos = end
            return result

        def calculate_semantic_boundaries(sentences, embedding_generator):
            import numpy as np

            if not embedding_generator or len(sentences) < 3:
                return []
            boundaries = []
            sentence_texts = [sent[0] for sent in sentences]
            logger.info("Generating embeddings for semantic chunking...")
            embeddings = embedding_generator.generate_embeddings_batch(sentence_texts)
            valid_embeddings = []
            valid_indices = []
            for i, emb in enumerate(embeddings):
                if emb is not None:
                    valid_embeddings.append(emb)
                    valid_indices.append(i)
            if len(valid_embeddings) < 3:
                logger.warning("Insufficient valid embeddings for semantic chunking")
                return boundaries
            similarities = []
            for i in range(len(valid_embeddings) - 1):
                emb1 = np.array(valid_embeddings[i])
                emb2 = np.array(valid_embeddings[i + 1])
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities.append(sim)
            threshold = np.mean(similarities) - 0.5 * np.std(similarities)
            for i in range(1, len(similarities) - 1):
                current_sim = similarities[i]
                prev_sim = similarities[i - 1]
                next_sim = similarities[i + 1]
                if (
                    current_sim < threshold
                    and current_sim < prev_sim
                    and current_sim < next_sim
                ):
                    sentence_idx = valid_indices[i + 1]
                    if sentence_idx < len(sentences):
                        position = sentences[sentence_idx][1]
                        confidence = 1.0 - current_sim
                        boundaries.append(
                            {
                                "position": position,
                                "confidence": confidence,
                            }
                        )
            return boundaries

        try:
            sentences = extract_sentences(text)
            if not sentences:
                return []
            boundaries = calculate_semantic_boundaries(
                sentences, self.embedding_generator
            )
            boundaries = sorted(boundaries, key=lambda b: b["position"])
            filtered_boundaries = []
            min_distance = self.min_chunk_size
            for boundary in boundaries:
                if boundary["confidence"] > 0.3:
                    if (
                        not filtered_boundaries
                        or boundary["position"] - filtered_boundaries[-1]["position"]
                        > min_distance
                    ):
                        filtered_boundaries.append(boundary)
            # Create chunks based on boundaries
            chunks = []
            start_pos = 0
            for boundary in filtered_boundaries:
                if boundary["position"] > start_pos + self.min_chunk_size:
                    chunk_text = text[start_pos : boundary["position"]].strip()
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    start_pos = boundary["position"]
            # Add final chunk
            if start_pos < len(text):
                remaining_text = text[start_pos:].strip()
                if len(remaining_text) >= self.min_chunk_size:
                    chunks.append(remaining_text)
            return chunks
        except Exception as e:
            logger.warning(
                f"Semantic chunking failed: {e}. Falling back to basic chunking."
            )
            return SentenceChunkingStrategy(
                self.chunk_size, self.min_chunk_size, self.overlap
            ).chunk(text)


class GradientChunkingStrategy(ChunkingStrategy):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Gradient chunking is not yet implemented.")

    def chunk(self, text: str) -> List[str]:
        pass


class AdaptiveChunkingStrategy(ChunkingStrategy):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Adaptive chunking is not yet implemented.")

    def chunk(self, text: str) -> List[str]:
        pass


class ChunkingStrategyFactory:
    _strategy_map: Dict[str, Type[ChunkingStrategy]] = {
        ChunkingMethod.SENTENCES: SentenceChunkingStrategy,
        ChunkingMethod.PARAGRAPHS: ParagraphChunkingStrategy,
        ChunkingMethod.SEMANTIC: SemanticChunkingStrategy,
        ChunkingMethod.GRADIENT: GradientChunkingStrategy,
        ChunkingMethod.ADAPTIVE: AdaptiveChunkingStrategy,
    }

    @classmethod
    def get_strategy(
        cls,
        method: str,
        chunk_size: int,
        min_chunk_size: int,
        overlap: int,
        embedding_generator=None,
    ) -> ChunkingStrategy:
        strategy_cls = cls._strategy_map.get(method, SentenceChunkingStrategy)
        return strategy_cls(chunk_size, min_chunk_size, overlap, embedding_generator)


class TextChunker:
    """Handles text chunking with configurable parameters and advanced methods using the Strategy pattern."""

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

    def chunk_text(
        self, text: str, method: str = ChunkingMethod.SENTENCES
    ) -> List[str]:
        strategy = ChunkingStrategyFactory.get_strategy(
            method,
            self.chunk_size,
            self.min_chunk_size,
            self.overlap,
            self.embedding_generator,
        )
        return strategy.chunk(text)
