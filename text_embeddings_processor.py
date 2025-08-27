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
from typing import Optional, Dict
from datetime import datetime
import hashlib
from src.config.config import Config
from src.storage.postgres import PostgresEmbeddingStorage
from src.chunking.text_chunker import TextChunker, ChunkingMethod
from src.embedding.ollama import OllamaEmbeddingGenerator


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

            # Chunk the text (pass method if available in metadata)
            method = None
            if metadata and "chunking_method" in metadata:
                method = metadata["chunking_method"]
            if method:
                chunks = self.chunker.chunk_text(content, method=method)
            else:
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
        choices=[
            ChunkingMethod.SENTENCES,
            ChunkingMethod.PARAGRAPHS,
            ChunkingMethod.SEMANTIC,
            ChunkingMethod.GRADIENT,
            ChunkingMethod.ADAPTIVE,
        ],
        default=ChunkingMethod.SENTENCES,
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
