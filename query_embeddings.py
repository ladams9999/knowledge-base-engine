#!/usr/bin/env python3
"""
Embeddings Query Utility

A utility script for querying and exploring stored embeddings in the PostgreSQL database.
Supports similarity search and document browsing.
"""

import argparse
import sys
from typing import List, Optional, Dict, Any
import logging

import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from src.config.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EmbeddingsQueryClient:
    """Client for querying stored embeddings."""

    def __init__(
        self,
        connection_params: Dict[str, Any],
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize query client.

        Args:
            connection_params: Database connection parameters
            ollama_base_url: Ollama server URL for generating query embeddings
        """
        self.connection_params = connection_params
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.connection = None

    def connect(self) -> bool:
        """Connect to database."""
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            logger.info("Connected to database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database."""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT d.*, COUNT(e.id) as chunk_count
                    FROM documents d
                    LEFT JOIN text_embeddings e ON d.id = e.document_id
                    GROUP BY d.id
                    ORDER BY d.created_at DESC
                """)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT chunk_index, text_content, text_hash, created_at
                    FROM text_embeddings
                    WHERE document_id = %s
                    ORDER BY chunk_index
                """,
                    (document_id,),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error retrieving document chunks: {e}")
            return []

    def generate_query_embedding(
        self, text: str, model: str = "nomic-embed-text"
    ) -> Optional[List[float]]:
        """Generate embedding for query text using Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("embedding")
            else:
                logger.error(
                    f"Failed to generate query embedding: {response.status_code}"
                )
                return None

        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return None

    def similarity_search(
        self, query_text: str, limit: int = 10, model: str = "nomic-embed-text"
    ) -> List[Dict[str, Any]]:
        """Perform similarity search against stored embeddings."""
        # Generate embedding for query
        query_embedding = self.generate_query_embedding(query_text, model)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []

        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT 
                        e.text_content,
                        e.chunk_index,
                        d.filename,
                        d.filepath,
                        1 - (e.embedding <=> %s::vector) as similarity_score
                    FROM text_embeddings e
                    JOIN documents d ON e.document_id = d.id
                    WHERE e.embedding IS NOT NULL
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                """,
                    (query_embedding, query_embedding, limit),
                )

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []

    def get_document_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT d.id) as document_count,
                        COUNT(e.id) as total_chunks,
                        COUNT(CASE WHEN e.embedding IS NOT NULL THEN 1 END) as embedded_chunks,
                        AVG(LENGTH(e.text_content)) as avg_chunk_length
                    FROM documents d
                    LEFT JOIN text_embeddings e ON d.id = e.document_id
                """)

                return dict(cursor.fetchone())

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()


def print_documents(documents: List[Dict[str, Any]]):
    """Print documents in a formatted table."""
    if not documents:
        print("No documents found.")
        return

    print(f"{'Filename':<30} {'Chunks':<8} {'Created':<20} {'ID':<36}")
    print("-" * 100)

    for doc in documents:
        filename = (
            doc["filename"][:29] if len(doc["filename"]) > 29 else doc["filename"]
        )
        created = (
            doc["created_at"].strftime("%Y-%m-%d %H:%M:%S")
            if doc["created_at"]
            else "Unknown"
        )
        print(f"{filename:<30} {doc['chunk_count']:<8} {created:<20} {doc['id']}")


def print_search_results(results: List[Dict[str, Any]]):
    """Print similarity search results."""
    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {result['similarity_score']:.4f}) ---")
        print(f"File: {result['filename']} (chunk {result['chunk_index']})")
        print(f"Text: {result['text_content'][:200]}...")


def print_chunks(chunks: List[Dict[str, Any]]):
    """Print document chunks."""
    if not chunks:
        print("No chunks found.")
        return

    for chunk in chunks:
        print(f"\n--- Chunk {chunk['chunk_index']} ---")
        print(f"Hash: {chunk['text_hash'][:16]}...")
        print(f"Length: {len(chunk['text_content'])} characters")
        print(f"Created: {chunk['created_at']}")
        print(f"Text preview: {chunk['text_content'][:150]}...")


def main():
    parser = argparse.ArgumentParser(description="Query embeddings database")
    parser.add_argument("config_file", help="Path to TOML config file")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List documents command
    subparsers.add_parser("list", help="List all documents")

    # Statistics command
    subparsers.add_parser("stats", help="Show database statistics")

    # Show chunks command
    chunks_parser = subparsers.add_parser("chunks", help="Show chunks for a document")
    chunks_parser.add_argument("document_id", help="Document ID")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar text")
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Load config using Config class
    config = Config(args.config_file)

    db_params = {
        "host": config.db_host,
        "port": config.db_port,
        "database": config.db_name,
        "user": config.db_user,
        "password": config.db_password,
    }

    # Initialize client
    client = EmbeddingsQueryClient(db_params, config.ollama_url)

    if not client.connect():
        return 1

    try:
        if args.command == "list":
            documents = client.list_documents()
            print_documents(documents)

        elif args.command == "stats":
            stats = client.get_document_stats()
            if stats:
                print("Database Statistics:")
                print(f"Documents: {stats['document_count']}")
                print(f"Total chunks: {stats['total_chunks']}")
                print(f"Embedded chunks: {stats['embedded_chunks']}")
                print(
                    f"Average chunk length: {stats['avg_chunk_length']:.1f} characters"
                )

        elif args.command == "chunks":
            chunks = client.get_document_chunks(args.document_id)
            print_chunks(chunks)

        elif args.command == "search":
            print(f"Searching for: '{args.query}'")
            results = client.similarity_search(args.query, args.limit, config.model)
            print_search_results(results)

        return 0

    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
