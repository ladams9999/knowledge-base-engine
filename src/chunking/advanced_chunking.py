#!/usr/bin/env python3
"""
Enhanced Text Chunking Strategies

Advanced chunking methods including semantic, gradient, and content-aware chunking
specifically optimized for conversational and speech content.
"""


import re
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging


# Optional imports for advanced features
try:
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning(
        "scikit-learn not available. Some advanced chunking methods will be limited."
    )

try:
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning(
        "spaCy not available. Named entity recognition features will be limited."
    )

logger = logging.getLogger(__name__)


@dataclass
class ChunkBoundary:
    """Represents a potential chunk boundary with confidence score."""

    position: int
    confidence: float
    reason: str
    boundary_type: str  # 'semantic', 'gradient', 'structural', 'hard'


@dataclass
class SemanticChunk:
    """Enhanced chunk with semantic metadata."""

    text: str
    start_pos: int
    end_pos: int
    semantic_score: float
    entities: List[str]
    keywords: List[str]
    topic_labels: List[str]


from src.chunking.text_chunker import ChunkingStrategy, ChunkingMethod

class AdvancedTextChunker(ChunkingStrategy):
    """Strategy for advanced chunking methods (semantic, gradient, adaptive, hierarchical)."""

    def __init__(
        self,
        chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap: int = 50,
        embedding_generator=None,
        max_chunk_size: int = 2000,
    ):
        super().__init__(chunk_size, min_chunk_size, overlap, embedding_generator)
        self.max_chunk_size = max_chunk_size

        # Initialize NLP components if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "spaCy English model not found. Install with: python -m spacy download en_core_web_sm"
                )
            except Exception:
                self.nlp = None

        # Discourse markers for conversation/speech content
        self.discourse_markers = [
            # Topic transitions
            r"now let['']?s talk about",
            r"moving on to",
            r"speaking of",
            r"that brings me to",
            r"on that note",
            r"while we['']?re on the subject",
            # Introductions
            r"today['']?s guest",
            r"i['']?m joined by",
            r"we have with us",
            r"let me introduce",
            # Segment markers
            r"before we get to that",
            r"first things first",
            r"real quick",
            r"one more thing",
            r"by the way",
            # Conclusions
            r"to wrap up",
            r"in conclusion",
            r"that['']?s all for",
            r"thanks for listening",
            r"before we go",
        ]
        self.discourse_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.discourse_markers
        ]
        self.content_markers = [
            r"afc\s+(?:east|west|north|south)",
            r"nfc\s+(?:east|west|north|south)",
            r"(?:week\s+\d+|season|playoffs|draft)",
            r"(?:quarterback|qb|running back|rb|wide receiver|wr)",
            r"(?:fantasy|rankings|projections)",
        ]
        self.content_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.content_markers
        ]

    def chunk(self, text: str) -> List[str]:
        # Default to semantic chunking for this strategy
        semantic_chunks = self.semantic_chunk_text(text)
        return [chunk.text for chunk in semantic_chunks]

    def extract_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract sentences with their positions in the text.

        Returns:
            List of (sentence_text, start_pos, end_pos) tuples
        """
        sentences = []

        if self.nlp:
            # Use spaCy for better sentence segmentation
            doc = self.nlp(text)
            for sent in doc.sents:
                sentences.append((sent.text.strip(), sent.start_char, sent.end_char))
        else:
            # Fallback to regex-based sentence splitting
            sentence_pattern = r"[.!?]+[\s]*"
            start = 0

            for match in re.finditer(sentence_pattern, text):
                end = match.end()
                sentence = text[start:end].strip()
                if len(sentence) > 10:  # Filter very short sentences
                    sentences.append((sentence, start, end))
                start = end

            # Add remaining text
            if start < len(text):
                remaining = text[start:].strip()
                if len(remaining) > 10:
                    sentences.append((remaining, start, len(text)))

        return sentences

    def calculate_semantic_boundaries(
        self, sentences: List[Tuple[str, int, int]]
    ) -> List[ChunkBoundary]:
        """
        Calculate semantic boundaries using embedding similarity.

        Args:
            sentences: List of (text, start_pos, end_pos) tuples

        Returns:
            List of potential chunk boundaries
        """
        if not self.embedding_generator or len(sentences) < 3:
            return []

        boundaries = []
        sentence_texts = [sent[0] for sent in sentences]

        # Generate embeddings for all sentences
        logger.info("Generating embeddings for semantic chunking...")
        embeddings = self.embedding_generator.generate_embeddings_batch(sentence_texts)

        # Filter out None embeddings
        valid_embeddings = []
        valid_indices = []
        for i, emb in enumerate(embeddings):
            if emb is not None:
                valid_embeddings.append(emb)
                valid_indices.append(i)

        if len(valid_embeddings) < 3:
            logger.warning("Insufficient valid embeddings for semantic chunking")
            return boundaries

        # Calculate similarity scores between adjacent sentences
        similarities = []
        for i in range(len(valid_embeddings) - 1):
            emb1 = np.array(valid_embeddings[i])
            emb2 = np.array(valid_embeddings[i + 1])

            # Cosine similarity
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(sim)

        # Find local minima in similarity (potential boundaries)
        threshold = np.mean(similarities) - 0.5 * np.std(similarities)

        for i in range(1, len(similarities) - 1):
            current_sim = similarities[i]
            prev_sim = similarities[i - 1]
            next_sim = similarities[i + 1]

            # Local minimum below threshold
            if (
                current_sim < threshold
                and current_sim < prev_sim
                and current_sim < next_sim
            ):
                sentence_idx = valid_indices[i + 1]  # Boundary after this sentence
                if sentence_idx < len(sentences):
                    position = sentences[sentence_idx][
                        1
                    ]  # Start position of next sentence
                    confidence = (
                        1.0 - current_sim
                    )  # Lower similarity = higher confidence

                    boundaries.append(
                        ChunkBoundary(
                            position=position,
                            confidence=confidence,
                            reason=f"Semantic boundary (similarity: {current_sim:.3f})",
                            boundary_type="semantic",
                        )
                    )

        return boundaries

    def calculate_gradient_boundaries(
        self, text: str, window_size: int = 3
    ) -> List[ChunkBoundary]:
        """
        Calculate boundaries using gradient-based analysis with rolling windows.

        Args:
            text: Input text
            window_size: Number of sentences in rolling window

        Returns:
            List of potential boundaries
        """
        sentences = self.extract_sentences(text)
        if len(sentences) < window_size * 2:
            return []

        boundaries = []

        # Use TF-IDF for quick similarity if embeddings aren't available
        if SKLEARN_AVAILABLE:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                sentence_texts = [sent[0] for sent in sentences]
                vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
                tfidf_matrix = vectorizer.fit_transform(sentence_texts)

                # Calculate similarity for overlapping windows
                gradients = []
                for i in range(len(sentences) - window_size):
                    window1_indices = list(range(i, i + window_size))
                    window2_indices = list(range(i + 1, i + window_size + 1))

                    # Average embeddings for each window
                    window1_vec = np.mean(
                        tfidf_matrix[window1_indices].toarray(), axis=0
                    )
                    window2_vec = np.mean(
                        tfidf_matrix[window2_indices].toarray(), axis=0
                    )

                    # Calculate similarity
                    similarity = cosine_similarity([window1_vec], [window2_vec])[0][0]
                    gradients.append(similarity)

                # Find steep drops in similarity (gradient boundaries)
                if len(gradients) > 2:
                    mean_grad = np.mean(gradients)
                    std_grad = np.std(gradients)
                    threshold = mean_grad - std_grad

                    for i, grad in enumerate(gradients):
                        if grad < threshold:
                            # Position at end of first window
                            boundary_sentence_idx = i + window_size
                            if boundary_sentence_idx < len(sentences):
                                position = sentences[boundary_sentence_idx][1]
                                confidence = (
                                    (mean_grad - grad) / std_grad
                                    if std_grad > 0
                                    else 0.5
                                )

                                boundaries.append(
                                    ChunkBoundary(
                                        position=position,
                                        confidence=min(confidence, 1.0),
                                        reason=f"Gradient boundary (similarity: {grad:.3f})",
                                        boundary_type="gradient",
                                    )
                                )
            except Exception as e:
                logger.warning(f"Error in gradient boundary calculation: {e}")

        return boundaries

    def find_structural_boundaries(self, text: str) -> List[ChunkBoundary]:
        """
        Find structural boundaries using discourse markers and content patterns.

        Args:
            text: Input text

        Returns:
            List of structural boundaries
        """
        boundaries = []

        # Find discourse markers
        for pattern in self.discourse_patterns:
            for match in pattern.finditer(text):
                boundaries.append(
                    ChunkBoundary(
                        position=match.start(),
                        confidence=0.8,
                        reason=f"Discourse marker: '{match.group()}'",
                        boundary_type="structural",
                    )
                )

        # Find content-specific markers
        for pattern in self.content_patterns:
            for match in pattern.finditer(text):
                boundaries.append(
                    ChunkBoundary(
                        position=match.start(),
                        confidence=0.6,
                        reason=f"Content marker: '{match.group()}'",
                        boundary_type="structural",
                    )
                )

        # Find paragraph breaks (hard boundaries)
        for match in re.finditer(r"\n\s*\n", text):
            boundaries.append(
                ChunkBoundary(
                    position=match.start(),
                    confidence=0.9,
                    reason="Paragraph break",
                    boundary_type="hard",
                )
            )

        return boundaries

    def extract_entities_and_keywords(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract named entities and keywords from text."""
        entities = []
        keywords = []

        if self.nlp:
            doc = self.nlp(text)
            entities = [ent.text for ent in doc.ents]
            keywords = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and len(token.text) > 2
            ]
        else:
            # Simple keyword extraction using content patterns
            for pattern in self.content_patterns:
                keywords.extend([match.group() for match in pattern.finditer(text)])

        return entities, keywords

    def semantic_chunk_text(self, text: str) -> List[SemanticChunk]:
        """
        Perform semantic chunking with enhanced metadata.

        Args:
            text: Input text to chunk

        Returns:
            List of semantic chunks with metadata
        """
        sentences = self.extract_sentences(text)
        if not sentences:
            return []

        # Find all types of boundaries
        semantic_boundaries = self.calculate_semantic_boundaries(sentences)
        gradient_boundaries = self.calculate_gradient_boundaries(text)
        structural_boundaries = self.find_structural_boundaries(text)

        # Combine and sort boundaries
        all_boundaries = (
            semantic_boundaries + gradient_boundaries + structural_boundaries
        )
        all_boundaries.sort(key=lambda b: b.position)

        # Filter boundaries by confidence and distance
        filtered_boundaries = []
        min_distance = self.min_chunk_size

        for boundary in all_boundaries:
            if boundary.confidence > 0.3:  # Confidence threshold
                # Check minimum distance from last boundary
                if (
                    not filtered_boundaries
                    or boundary.position - filtered_boundaries[-1].position
                    > min_distance
                ):
                    filtered_boundaries.append(boundary)

        # Create chunks based on boundaries
        chunks = []
        start_pos = 0

        for boundary in filtered_boundaries:
            if boundary.position > start_pos + self.min_chunk_size:
                chunk_text = text[start_pos : boundary.position].strip()

                if len(chunk_text) >= self.min_chunk_size:
                    entities, keywords = self.extract_entities_and_keywords(chunk_text)

                    chunks.append(
                        SemanticChunk(
                            text=chunk_text,
                            start_pos=start_pos,
                            end_pos=boundary.position,
                            semantic_score=boundary.confidence,
                            entities=entities,
                            keywords=keywords,
                            topic_labels=[],  # Could be enhanced with topic modeling
                        )
                    )

                    start_pos = boundary.position

        # Add final chunk
        if start_pos < len(text):
            remaining_text = text[start_pos:].strip()
            if len(remaining_text) >= self.min_chunk_size:
                entities, keywords = self.extract_entities_and_keywords(remaining_text)

                chunks.append(
                    SemanticChunk(
                        text=remaining_text,
                        start_pos=start_pos,
                        end_pos=len(text),
                        semantic_score=0.5,  # Default score
                        entities=entities,
                        keywords=keywords,
                        topic_labels=[],
                    )
                )

        logger.info(
            f"Created {len(chunks)} semantic chunks with average size {np.mean([len(c.text) for c in chunks]):.0f} characters"
        )

        return chunks

    def hierarchical_chunk_text(self, text: str) -> Dict[str, List[SemanticChunk]]:
        """
        Create hierarchical chunks at different granularity levels.

        Returns:
            Dictionary with different chunk granularities: 'fine', 'medium', 'coarse'
        """
        # Fine-grained chunks (high boundary confidence)
        fine_chunker = AdvancedTextChunker(
            embedding_generator=self.embedding_generator,
            chunk_size=self.chunk_size // 2,
            min_chunk_size=self.min_chunk_size // 2,
            max_chunk_size=self.chunk_size,
        )

        # Coarse-grained chunks (low boundary confidence, larger sizes)
        coarse_chunker = AdvancedTextChunker(
            embedding_generator=self.embedding_generator,
            chunk_size=self.chunk_size * 2,
            min_chunk_size=self.chunk_size,
            max_chunk_size=self.chunk_size * 3,
        )

        return {
            "fine": fine_chunker.semantic_chunk_text(text),
            "medium": self.semantic_chunk_text(text),
            "coarse": coarse_chunker.semantic_chunk_text(text),
        }

    def chunk_text_adaptive(self, text: str, method: str = ChunkingMethod.SEMANTIC) -> List[str]:
        """
        Main chunking method with adaptive strategy selection.
        Args:
            text: Input text
            method: Chunking method (use ChunkingMethod constants)
        Returns:
            List of text chunks (strings for compatibility)
        """
        if method == ChunkingMethod.SEMANTIC:
            semantic_chunks = self.semantic_chunk_text(text)
            return [chunk.text for chunk in semantic_chunks]
        elif method == ChunkingMethod.GRADIENT:
            # For demonstration, use semantic chunking as a fallback
            semantic_chunks = self.semantic_chunk_text(text)
            return [chunk.text for chunk in semantic_chunks]
        elif method == ChunkingMethod.ADAPTIVE:
            # For demonstration, use semantic chunking as a fallback
            semantic_chunks = self.semantic_chunk_text(text)
            return [chunk.text for chunk in semantic_chunks]
        else:
            logger.warning(f"Falling back to basic chunking for method: {method}")
            sentences = self.extract_sentences(text)
            chunks = []
            current_chunk = ""
            for sentence, _, _ in sentences:
                if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                    if len(current_chunk) >= self.min_chunk_size:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            if current_chunk and len(current_chunk) >= self.min_chunk_size:
                chunks.append(current_chunk.strip())
            return chunks
