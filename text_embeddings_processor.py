#!/usr/bin/env python3
"""
Text Embeddings Processor

A comprehensive tool for processing text files by chunking, generating embeddings using Ollama,
and storing them in PostgreSQL database with proper error handling and logging.
"""

import argparse
import logging
import sys
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
from src.config.config import Config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("embeddings_processor.log"),
    ],
)
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
        """
        Initialize TextChunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be considered valid
            embedding_generator: Ollama embedding generator for semantic chunking
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.embedding_generator = embedding_generator

        # Initialize advanced chunker if needed
        self._advanced_chunker = None

    def _get_advanced_chunker(self):
        """Lazy initialization of advanced chunker."""
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
        """
        Chunk text by sentences, respecting chunk size limits.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        # Split by common sentence endings
        sentences = []
        current_sentence = ""

        for char in text:
            current_sentence += char
            if char in ".!?" and len(current_sentence.strip()) > 10:
                sentences.append(current_sentence.strip())
                current_sentence = ""

        # Add remaining text as last sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed chunk size, finalize current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add the final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Chunk text by paragraphs, splitting large paragraphs if needed.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
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
                # Split large paragraphs by sentences
                sentence_chunks = self.chunk_by_sentences(paragraph)
                chunks.extend(sentence_chunks)

        return chunks

    def chunk_text(self, text: str, method: str = "sentences") -> List[str]:
        """
        Chunk text using specified method.

        Args:
            text: Input text to chunk
            method: Chunking method ("sentences", "paragraphs", "semantic", "gradient", "adaptive")

        Returns:
            List of text chunks
        """
        # Check for advanced methods
        if method in ["semantic", "gradient", "adaptive", "hierarchical"]:
            advanced_chunker = self._get_advanced_chunker()
            if advanced_chunker:
                try:
                    return advanced_chunker.chunk_text_adaptive(text, method)
                except Exception as e:
                    logger.warning(
                        f"Advanced chunking failed ({method}): {e}. Falling back to basic chunking."
                    )
                    method = "sentences"  # Fallback
            else:
                logger.warning(
                    f"Advanced chunking not available. Falling back to basic method."
                )
                method = "sentences"  # Fallback

        # Basic chunking methods
        if method == "paragraphs":
            return self.chunk_by_paragraphs(text)
        else:
            return self.chunk_by_sentences(text)


class OllamaEmbeddingGenerator:
    """Handles embedding generation using Ollama API."""

    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"
    ):
        """
        Initialize Ollama embedding generator.

        Args:
            base_url: Ollama server base URL
            model: Embedding model name
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = requests.Session()

    def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            return False

    def ensure_model_available(self) -> bool:
        """Ensure the embedding model is available."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]

                if self.model in available_models:
                    return True
                else:
                    logger.warning(
                        f"Model {self.model} not found. Available models: {available_models}"
                    )
                    logger.info(f"Pulling model {self.model}...")
                    return self.pull_model()
            return False
        except Exception as e:
            logger.error(f"Error checking available models: {e}")
            return False

    def pull_model(self) -> bool:
        """Pull the embedding model if not available."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/pull", json={"name": self.model}, timeout=300
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error pulling model {self.model}: {e}")
            return False

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats, or None if failed
        """
        try:
            payload = {"model": self.model, "prompt": text}

            response = self.session.post(
                f"{self.base_url}/api/embeddings", json=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("embedding")
            else:
                logger.error(
                    f"Embedding generation failed: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def generate_embeddings_batch(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for i, text in enumerate(texts):
            logger.info(f"Generating embedding {i + 1}/{len(texts)}")
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings


class PostgresEmbeddingStorage:
    """Handles storage of embeddings in PostgreSQL database."""

    def __init__(self, connection_params: Dict[str, Any]):
        """
        Initialize PostgreSQL storage.

        Args:
            connection_params: Database connection parameters
        """
        self.connection_params = connection_params
        self.connection = None

    def connect(self) -> bool:
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            logger.info("Successfully connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def create_tables(self) -> bool:
        """Create necessary tables if they don't exist."""
        try:
            with self.connection.cursor() as cursor:
                # Create documents table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        filename VARCHAR(255) NOT NULL,
                        filepath VARCHAR(500) NOT NULL,
                        content_hash VARCHAR(64) NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Create embeddings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS text_embeddings (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        text_content TEXT NOT NULL,
                        embedding VECTOR(768),  -- Adjust dimension based on your model
                        text_hash VARCHAR(64) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(document_id, chunk_index)
                    );
                """)

                # Create indexes
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON text_embeddings(document_id);"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_embeddings_text_hash ON text_embeddings(text_hash);"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);"
                )

                self.connection.commit()
                logger.info("Database tables created/verified successfully")
                return True

        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            self.connection.rollback()
            return False

    def get_or_create_document(
        self, filepath: str, content_hash: str, metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Get existing document or create new one.

        Args:
            filepath: Path to the source file
            content_hash: Hash of file content
            metadata: Additional metadata

        Returns:
            Document ID as string, or None if failed
        """
        try:
            filename = os.path.basename(filepath)

            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Check if document already exists
                cursor.execute(
                    "SELECT id FROM documents WHERE filepath = %s AND content_hash = %s",
                    (filepath, content_hash),
                )
                result = cursor.fetchone()

                if result:
                    return str(result["id"])

                # Create new document
                cursor.execute(
                    """
                    INSERT INTO documents (filename, filepath, content_hash, metadata)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """,
                    (filename, filepath, content_hash, json.dumps(metadata or {})),
                )

                result = cursor.fetchone()
                self.connection.commit()
                return str(result["id"])

        except Exception as e:
            logger.error(f"Error creating/retrieving document: {e}")
            self.connection.rollback()
            return None

    def store_embeddings(
        self,
        document_id: str,
        chunks: List[str],
        embeddings: List[Optional[List[float]]],
    ) -> bool:
        """
        Store text chunks and their embeddings.

        Args:
            document_id: Document ID
            chunks: List of text chunks
            embeddings: List of corresponding embeddings

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.connection.cursor() as cursor:
                # Clear existing embeddings for this document
                cursor.execute(
                    "DELETE FROM text_embeddings WHERE document_id = %s", (document_id,)
                )

                # Insert new embeddings
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    if embedding is None:
                        logger.warning(f"Skipping chunk {i} due to missing embedding")
                        continue

                    text_hash = hashlib.sha256(chunk.encode()).hexdigest()

                    cursor.execute(
                        """
                        INSERT INTO text_embeddings 
                        (document_id, chunk_index, text_content, embedding, text_hash)
                        VALUES (%s, %s, %s, %s, %s)
                    """,
                        (document_id, i, chunk, embedding, text_hash),
                    )

                self.connection.commit()
                logger.info(
                    f"Stored {len([e for e in embeddings if e is not None])} embeddings for document {document_id}"
                )
                return True

        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            self.connection.rollback()
            return False

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


class TextEmbeddingsProcessor:
    """Main processor class that orchestrates the entire pipeline."""

    def __init__(
        self,
        chunker: TextChunker,
        embedding_generator: OllamaEmbeddingGenerator,
        storage: PostgresEmbeddingStorage,
    ):
        """
        Initialize the processor.

        Args:
            chunker: Text chunking handler
            embedding_generator: Embedding generation handler
            storage: Database storage handler
        """
        self.chunker = chunker
        self.embedding_generator = embedding_generator
        self.storage = storage

    def process_file(self, filepath: str, metadata: Optional[Dict] = None) -> bool:
        """
        Process a single text file.

        Args:
            filepath: Path to text file
            metadata: Additional metadata for the document

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read file content
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            logger.info(f"Processing file: {filepath}")
            logger.info(f"File size: {len(content)} characters")

            # Calculate content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Get or create document record
            document_id = self.storage.get_or_create_document(
                filepath, content_hash, metadata
            )
            if not document_id:
                logger.error(f"Failed to create document record for {filepath}")
                return False

            # Chunk the text
            chunks = self.chunker.chunk_text(content)
            logger.info(f"Created {len(chunks)} chunks")

            if not chunks:
                logger.warning(f"No valid chunks created for {filepath}")
                return False

            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_generator.generate_embeddings_batch(chunks)

            # Store embeddings
            success = self.storage.store_embeddings(document_id, chunks, embeddings)

            if success:
                logger.info(f"Successfully processed {filepath}")
            else:
                logger.error(f"Failed to store embeddings for {filepath}")

            return success

        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")
            return False


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file {config_file}: {e}")
        return {}


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Process text files and generate embeddings"
    )
    parser.add_argument("config_file", help="Path to TOML config file")
    parser.add_argument("input_file", help="Path to input text file")
    parser.add_argument("--chunk-size", type=int, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, help="Overlap between chunks")
    parser.add_argument(
        "--method",
        choices=["sentences", "paragraphs", "semantic", "gradient", "adaptive"],
        default="sentences",
        help="Chunking method",
    )

    args = parser.parse_args()

    # Load config from TOML file
    config = Config(args.config_file)

    # Validate input file
    if not os.path.exists(args.input_file):
        logger.error(f"Input file does not exist: {args.input_file}")
        return 1

    # Use CLI overrides if provided, else config values
    chunk_size = args.chunk_size if args.chunk_size is not None else 1000
    overlap = args.overlap if args.overlap is not None else 100
    min_chunk_size = 50  # Could be made configurable if needed

    # Initialize components
    embedding_generator = OllamaEmbeddingGenerator(
        base_url=config.ollama_url, model=config.model
    )

    # Test Ollama connection
    if not embedding_generator.test_connection():
        logger.error("Failed to connect to Ollama server")
        return 1

    # Ensure model is available
    if not embedding_generator.ensure_model_available():
        logger.error(f"Model {config.model} is not available")
        return 1

    chunker = TextChunker(
        chunk_size=chunk_size,
        overlap=overlap,
        min_chunk_size=min_chunk_size,
        embedding_generator=embedding_generator,
    )

    # Database connection parameters
    db_params = {
        "host": config.db_host,
        "port": config.db_port,
        "database": config.db_name,
        "user": config.db_user,
        "password": config.db_password,
    }

    storage = PostgresEmbeddingStorage(db_params)

    # Connect to database
    if not storage.connect():
        logger.error("Failed to connect to database")
        return 1

    # Create tables
    if not storage.create_tables():
        logger.error("Failed to create database tables")
        return 1

    # Initialize processor
    processor = TextEmbeddingsProcessor(chunker, embedding_generator, storage)

    # Process file
    metadata = {
        "processing_date": datetime.now().isoformat(),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunking_method": args.method,
        "embedding_model": config.model,
    }

    try:
        success = processor.process_file(args.input_file, metadata)
        if success:
            logger.info("Processing completed successfully")
            return 0
        else:
            logger.error("Processing failed")
            return 1
    finally:
        storage.close()


if __name__ == "__main__":
    sys.exit(main())
