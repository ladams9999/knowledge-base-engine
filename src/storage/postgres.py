import os
import json
import hashlib
from typing import List, Optional, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor

import logging

logger = logging.getLogger(__name__)


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
