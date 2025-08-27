# WIP - Knowledge Base Engine

Advanced text processing and embeddings system with semantic chunking, PostgreSQL storage, and Ollama integration.

## Features

🧠 **Advanced Text Chunking**
- Semantic chunking that preserves topical coherence
- Gradient-based chunking using statistical analysis
- Adaptive chunking with automatic method selection
- Traditional sentence and paragraph chunking

🔍 **Intelligent Text Processing**
- Embedding generation via Ollama integration
- Vector similarity search with pgvector
- Named entity recognition with spaCy
- Content-specific discourse marker detection

📊 **PostgreSQL Storage**
- Efficient vector storage with pgvector extension
- Document management with metadata tracking
- Similarity search with cosine distance
- Comprehensive indexing for fast queries

🎯 **Multiple Use Cases**
- Knowledge base construction
- Semantic document search
- Content analysis and organization
- Conversational AI data preparation

## Installation

### Prerequisites

1. **Python 3.13+** and **uv** package manager
2. **PostgreSQL** with **pgvector** extension
3. **Ollama** server running locally

### Setup

1. Clone and install dependencies:
```bash
cd knowledge-base-engine
uv sync
```

2. Install spaCy English model:
```bash
uv run python -m spacy download en_core_web_sm
```

3. Set up PostgreSQL database:
```sql
CREATE DATABASE knowledge_base;
\c knowledge_base
CREATE EXTENSION vector;
```

4. Start Ollama and pull embedding model:
```bash
ollama serve
ollama pull nomic-embed-text
```

## Usage

### Basic Text Processing

Process a text file with semantic chunking:

```bash
uv run python text_embeddings_processor.py sample.txt \
  --method semantic \
  --chunk-size 1000 \
  --db-name knowledge_base \
  --db-user postgres
```

### Available Chunking Methods

- `sentences` - Traditional sentence-based chunking
- `paragraphs` - Paragraph-based chunking
- `semantic` - 🧠 Semantic chunking (recommended)
- `gradient` - Statistical gradient analysis
- `adaptive` - 🎯 Automatic method selection

### Querying the Database

List all documents:
```bash
uv run python query_embeddings.py list --db-name knowledge_base --db-user postgres
```

Search for similar content:
```bash
uv run python query_embeddings.py search "machine learning concepts" \
  --db-name knowledge_base --db-user postgres --limit 5
```

View document statistics:
```bash
uv run python query_embeddings.py stats --db-name knowledge_base --db-user postgres
```

### Testing the System

Run the comprehensive test suite:
```bash
uv run python test_embeddings.py
```

Run the chunking demonstration:
```bash
uv run python chunking_demo.py
```

## Advanced Features

### Semantic Chunking

Semantic chunking analyzes embedding similarity to find natural breakpoints:

```python
from text_embeddings_processor import TextChunker
from text_embeddings_processor import OllamaEmbeddingGenerator

# Initialize with embedding generator for semantic analysis
embedding_gen = OllamaEmbeddingGenerator()
chunker = TextChunker(
    chunk_size=1000,
    min_chunk_size=200,
    embedding_generator=embedding_gen
)

# Use semantic chunking
chunks = chunker.chunk_text(content, method="semantic")
```

### Hierarchical Chunking

Generate chunks at multiple granularity levels:

```python
from advanced_chunking import AdvancedTextChunker

chunker = AdvancedTextChunker(embedding_generator=embedding_gen)
hierarchical = chunker.hierarchical_chunk_text(content)

# Access different levels
fine_chunks = hierarchical['fine']      # High detail
medium_chunks = hierarchical['medium']  # Balanced
coarse_chunks = hierarchical['coarse']  # High level
```

### Custom Content Patterns

The system includes specialized patterns for different content types:

```python
# Sports/football content patterns
content_markers = [
    r"afc\s+(?:east|west|north|south)",
    r"nfc\s+(?:east|west|north|south)",
    r"(?:quarterback|qb|running back|rb)",
]

# Conversation/speech patterns
discourse_markers = [
    r"now let['']?s talk about",
    r"moving on to",
    r"speaking of",
    r"that brings me to",
]
```

## Configuration

### Environment Variables

```bash
export POSTGRES_PASSWORD=your_password
export OLLAMA_BASE_URL=http://localhost:11434
```

### Command Line Options

**Text Processing:**
- `--chunk-size`: Target chunk size in characters (default: 1000)
- `--overlap`: Overlap between chunks (default: 100)
- `--method`: Chunking method (default: sentences)
- `--model`: Ollama embedding model (default: nomic-embed-text)

**Database:**
- `--db-host`: Database host (default: localhost)
- `--db-port`: Database port (default: 5432)
- `--db-name`: Database name (required)
- `--db-user`: Database user (required)
- `--db-password`: Database password (or use env var)

## Architecture

### Core Components

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Text Input        │───▶│  Advanced Chunker    │───▶│  Embedding Generator│
│   - Documents       │    │  - Semantic          │    │  - Ollama API       │
│   - Content files   │    │  - Gradient          │    │  - Vector creation  │
└─────────────────────┘    │  - Adaptive          │    └─────────────────────┘
                           └──────────────────────┘               │
                                      │                           │
                                      ▼                           ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Query Interface   │◀───│  PostgreSQL Storage  │◀───│  Processing Pipeline│
│   - Similarity      │    │  - pgvector          │    │  - Chunking         │
│   - Document browse │    │  - Metadata          │    │  - Embedding        │
│   - Statistics      │    │  - Indexing          │    │  - Storage          │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### Key Classes

- **`TextEmbeddingsProcessor`** - Main orchestration class
- **`AdvancedTextChunker`** - Semantic and gradient chunking
- **`OllamaEmbeddingGenerator`** - Embedding generation via Ollama
- **`PostgresEmbeddingStorage`** - Database operations
- **`EmbeddingsQueryClient`** - Search and retrieval

## Benefits Over Traditional Approaches

### Semantic Chunking Advantages

✅ **Better Context Preservation**
- Keeps related topics together
- Maintains coherence across chunk boundaries
- Reduces information fragmentation

✅ **Improved Search Quality**
- More meaningful embedding representations
- Better similarity match accuracy
- Reduced noise in search results

✅ **Content-Aware Processing**
- Recognizes discourse markers
- Handles conversational content
- Adapts to content structure

### Performance Benefits

- **Efficient Vector Operations**: pgvector provides optimized similarity search
- **Batch Processing**: Ollama integration supports batch embedding generation
- **Scalable Architecture**: Handles large document collections
- **Intelligent Caching**: Avoids reprocessing unchanged content

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure all dependencies are installed
uv sync

# Install spaCy model
uv run python -m spacy download en_core_web_sm
```

**Database Connection Issues:**
```sql
-- Verify pgvector extension
\dx vector

-- Create extension if missing
CREATE EXTENSION vector;
```

**Ollama Connection Problems:**
```bash
# Start Ollama server
ollama serve

# Verify model is available
ollama list

# Pull embedding model if missing
ollama pull nomic-embed-text
```

### Performance Tuning

**For Large Documents:**
- Use `--method adaptive` for automatic optimization
- Increase `--chunk-size` for better context
- Enable batch processing for multiple files

**For Better Accuracy:**
- Use `--method semantic` for coherent chunking
- Adjust `--overlap` for smoother boundaries
- Fine-tune similarity thresholds

## Contributing

The system is designed for extensibility:

1. **Custom Chunking Methods**: Extend `AdvancedTextChunker`
2. **New Content Patterns**: Add domain-specific markers
3. **Alternative Embeddings**: Support additional providers
4. **Enhanced Queries**: Extend `EmbeddingsQueryClient`

## License

This project combines and enhances text processing capabilities from the flotsam-jetsam project, adapted for use with UV package management and modern Python practices.

---

🚀 **Ready to process your knowledge base with intelligent text chunking!**

Run the demonstration to see the system in action:
```bash
uv run python chunking_demo.py
```
