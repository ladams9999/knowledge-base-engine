#!/usr/bin/env python3
"""
Test script for Text Embeddings Processor

This script runs basic tests to ensure the embeddings processor is working correctly.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

def create_test_text_file():
    """Create a temporary test text file."""
    test_content = """
    This is a test document for the Text Embeddings Processor.
    
    The processor should be able to chunk this text into smaller pieces,
    generate embeddings for each chunk using Ollama, and store them in
    a PostgreSQL database.
    
    This paragraph contains some specific information about technology.
    Machine learning models can process natural language and convert
    text into numerical representations called embeddings. These
    embeddings capture semantic meaning and can be used for similarity search.
    
    The pgvector extension for PostgreSQL provides efficient storage
    and querying capabilities for high-dimensional vectors.
    
    Advanced chunking methods like semantic chunking can preserve
    topical coherence by analyzing the meaning of sentences and
    finding natural breakpoints in the text.
    """
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content.strip())
        return f.name


def test_chunking():
    """Test the text chunking functionality."""
    print("Testing text chunking...")
    
    try:
        from text_embeddings_processor import TextChunker
        
        chunker = TextChunker(chunk_size=200, overlap=50)
        test_text = "This is sentence one. This is sentence two! This is sentence three? " * 10
        
        chunks = chunker.chunk_text(test_text)
        
        if len(chunks) > 1:
            print(f"✓ Chunking successful: {len(chunks)} chunks created")
            return True
        else:
            print("✗ Chunking failed: Only one chunk created")
            return False
            
    except Exception as e:
        print(f"✗ Chunking test failed: {e}")
        return False


def test_ollama_connection(ollama_url="http://localhost:11434"):
    """Test connection to Ollama server."""
    print("Testing Ollama connection...")
    
    try:
        from text_embeddings_processor import OllamaEmbeddingGenerator
        
        generator = OllamaEmbeddingGenerator(base_url=ollama_url)
        
        if generator.test_connection():
            print("✓ Ollama connection successful")
            return True
        else:
            print("✗ Failed to connect to Ollama server")
            print(f"  Make sure Ollama is running: ollama serve")
            return False
            
    except Exception as e:
        print(f"✗ Ollama connection test failed: {e}")
        return False


def test_database_connection(db_params):
    """Test PostgreSQL database connection."""
    print("Testing database connection...")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()
        cursor.close()
        conn.close()
        
        print(f"✓ Database connection successful")
        print(f"  PostgreSQL version: {version[0].split()[1]}")
        return True
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


def test_pgvector_extension(db_params):
    """Test pgvector extension availability."""
    print("Testing pgvector extension...")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        # Check if pgvector extension is available
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            print("✓ pgvector extension is available")
            return True
        else:
            print("✗ pgvector extension not found")
            print("  Install with: CREATE EXTENSION vector;")
            return False
            
    except Exception as e:
        print(f"✗ pgvector test failed: {e}")
        return False


def run_integration_test(test_file, db_params):
    """Run a full integration test."""
    print("\nRunning integration test...")
    
    script_dir = Path(__file__).parent
    processor_script = script_dir / "text_embeddings_processor.py"
    
    cmd = [
        sys.executable,  # Use current Python interpreter
        str(processor_script),
        test_file,
        "--db-name", db_params['database'],
        "--db-user", db_params['user'],
        "--db-password", db_params['password'],
        "--chunk-size", "150",
        "--method", "sentences",
        "--model", "nomic-embed-text"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Integration test successful")
            print("  Text file processed and embeddings stored")
            return True
        else:
            print(f"✗ Integration test failed (exit code: {result.returncode})")
            print(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Integration test timed out")
        return False
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def test_advanced_chunking():
    """Test advanced chunking functionality."""
    print("Testing advanced chunking...")
    
    try:
        from text_embeddings_processor import TextChunker
        from advanced_chunking import AdvancedTextChunker
        
        # Mock embedding generator for testing
        class MockEmbeddingGenerator:
            def generate_embeddings_batch(self, texts):
                import hashlib
                return [[float(i % 100) / 100 for i in range(768)] for _ in texts]
        
        mock_embedder = MockEmbeddingGenerator()
        chunker = TextChunker(
            chunk_size=300,
            min_chunk_size=100,
            embedding_generator=mock_embedder
        )
        
        test_text = """
        This is the introduction to our test document.
        We're testing semantic chunking capabilities.
        
        Now let's talk about technology.
        Machine learning and AI are transforming many industries.
        Natural language processing is a key component.
        
        Moving on to databases.
        PostgreSQL is a powerful relational database.
        Vector storage enables similarity search.
        """
        
        # Test semantic chunking
        chunks = chunker.chunk_text(test_text, method="semantic")
        
        if len(chunks) > 0:
            print(f"✓ Advanced chunking successful: {len(chunks)} semantic chunks created")
            return True
        else:
            print("✗ Advanced chunking failed: No chunks created")
            return False
            
    except ImportError:
        print("✗ Advanced chunking modules not available")
        return False
    except Exception as e:
        print(f"✗ Advanced chunking test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Knowledge Base Engine - Text Embeddings Processor Test Suite")
    print("=" * 65)
    
    # Get database parameters from command line or environment
    db_name = input("Enter database name [test_embeddings]: ") or "test_embeddings"
    db_user = input("Enter database username [postgres]: ") or "postgres"
    db_password = os.getenv('POSTGRES_PASSWORD')
    if not db_password:
        import getpass
        db_password = getpass.getpass("Enter database password: ")
    
    db_params = {
        'host': 'localhost',
        'port': 5432,
        'database': db_name,
        'user': db_user,
        'password': db_password
    }
    
    tests = [
        ("Chunking", lambda: test_chunking()),
        ("Advanced Chunking", lambda: test_advanced_chunking()),
        ("Ollama Connection", lambda: test_ollama_connection()),
        ("Database Connection", lambda: test_database_connection(db_params)),
        ("pgvector Extension", lambda: test_pgvector_extension(db_params)),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
    
    print(f"\n{'='*65}")
    print(f"Basic tests passed: {passed}/{total}")
    
    # Run integration test if basic tests pass
    if passed == total:
        test_file = create_test_text_file()
        try:
            if run_integration_test(test_file, db_params):
                print("\n🎉 All tests passed! The embeddings processor is ready to use.")
                print("\nUsage examples:")
                print("  # Process a file with semantic chunking")
                print(f"  uv run python text_embeddings_processor.py sample.txt --method semantic \\")
                print(f"    --db-name {db_name} --db-user {db_user}")
                print("\n  # Search embeddings database")
                print(f"  uv run python query_embeddings.py search 'your query' \\")
                print(f"    --db-name {db_name} --db-user {db_user}")
                print("\n  # Run chunking demonstration")
                print("  uv run python chunking_demo.py")
                return 0
            else:
                print("\n❌ Integration test failed.")
                return 1
        finally:
            # Clean up test file
            os.unlink(test_file)
    else:
        print(f"\n❌ {total - passed} basic tests failed. Fix these issues before running integration test.")
        print("\nTroubleshooting:")
        print("- Make sure all dependencies are installed: uv sync")
        print("- Ensure Ollama is running: ollama serve")
        print("- Verify PostgreSQL is accessible and pgvector is installed")
        print("- Check database credentials")
        return 1


if __name__ == "__main__":
    sys.exit(main())
