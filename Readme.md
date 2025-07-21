# Vectolite

A lightweight vector database using SQLite for storage. Fast, simple, and portable - perfect for local AI applications, document search, and semantic similarity tasks.

## Features

- **Local-first**: No external dependencies for basic functionality
- **Fast**: Efficient cosine similarity search with SQLite
- **Flexible Embeddings**: Support for both local (HuggingFace) and cloud (OpenAI) models
- **File Ingestion**: Direct support for .txt and .md files with smart chunking
- **Full CRUD**: Add, query, list, show, and delete documents
- **Statistics**: Track your database size and document count
- **CLI Ready**: Complete command-line interface included

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

```bash
# Add a document
python vectolite.py add "Machine learning is transforming how we process data"

# Query for similar content
python vectolite.py query "AI and data processing"

# Ingest a file
python vectolite.py ingest-file ./notes.md

# List all documents
python vectolite.py list

# Show database stats
python vectolite.py stats
```

### UI (Experimental)
```bash
streamlit run vectolite_ui.py
```

### Python API

```python
from vectolite import Vectolite, resolve_embed_fn

# Initialize with local embeddings (default)
embed_fn = resolve_embed_fn("all-MiniLM-L6-v2", local=True)
db = Vectolite("my_database.db", embed_fn=embed_fn)

# Add documents
db.insert("Local embeddings are fast and free.")
db.insert("Vector databases enable semantic search.", 
          metadata={"source": "docs", "topic": "search"})

# Query for similar content
results = db.query("Tell me about local models.", top_k=3)
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text']}")
```

## Advance CLI Commands

### Document Management

```bash
# Add a single document
python vectolite.py add "Your text here" --metadata '{"author": "you"}'

# Add with remote embeddings
python vectolite.py add "Hello world" --remote --model text-embedding-3-small

# Ingest files with chunking
python vectolite.py ingest-file ./document.txt --chunk --max-chars 1500

# Ingest without chunking
python vectolite.py ingest-file ./notes.md --no-chunk
```

### Search & Browse

```bash
# Query documents
python vectolite.py query "machine learning" --top-k 5

# List documents (paginated)
python vectolite.py list --limit 20 --offset 10

# List without showing text
python vectolite.py list --no-text

# Show a specific document
python vectolite.py show 42
```

### Database Operations

```bash
# Show database statistics
python vectolite.py stats

# Delete a document
python vectolite.py delete 123

# Use custom database file
python vectolite.py query "search term" --db ./custom.db
```

## Model Selection

### Local Models (HuggingFace)

```bash
# Fast and lightweight (default)
python vectolite.py add "text" --local --model all-MiniLM-L6-v2

# Higher quality, larger size
python vectolite.py add "text" --local --model all-mpnet-base-v2

# Multilingual support
python vectolite.py add "text" --local --model paraphrase-multilingual-MiniLM-L12-v2
```

### Remote Models (OpenAI)

```bash
# Requires OPENAI_API_KEY environment variable
export OPENAI_API_KEY="your-key-here"

# Small model (cheaper)
python vectolite.py add "text" --remote --model text-embedding-3-small

# Large model (higher quality)
python vectolite.py add "text" --remote --model text-embedding-3-large
```

## Advanced Usage

### File Ingestion with Custom Chunking

```bash
# Custom chunk size and overlap
python vectolite.py ingest-file large_document.txt \
  --chunk \
  --max-chars 2000 \
  --overlap 300 \
  --metadata '{"document_type": "research"}'
```

### Batch Operations

```bash
# Ingest multiple files
for file in *.md; do
  python vectolite.py ingest-file "$file" --metadata "{\"filename\": \"$file\"}"
done
```

### Python API Advanced Examples

```python
from vectolite import Vectolite, resolve_embed_fn, chunk_text

# Initialize with OpenAI embeddings
embed_fn = resolve_embed_fn("text-embedding-3-small", local=False)
db = Vectolite("openai_database.db", embed_fn=embed_fn)

# Custom text chunking
text = "Long document content..."
chunks = chunk_text(text, max_chars=1000, overlap=100)

for i, chunk in enumerate(chunks):
    db.insert(chunk, metadata={"chunk_id": i, "total_chunks": len(chunks)})

# Advanced querying
results = db.query("search term", top_k=10)

# Database management
print(f"Total documents: {db.count_documents()}")
documents = db.list_documents(limit=5, include_text=False)
```

## Configuration

### Environment Variables

```bash
# For OpenAI embeddings
export OPENAI_API_KEY="your-api-key"

# Optional: Set default database location
export VECTOLITE_DB="./my_default.db"
```

### Recommended Models

| Use Case | Model | Type | Size |
|----------|--------|------|------|
| General purpose (fast) | `all-MiniLM-L6-v2` | Local | 23MB |
| High quality | `all-mpnet-base-v2` | Local | 438MB |
| Multilingual | `paraphrase-multilingual-MiniLM-L12-v2` | Local | 278MB |
| Production (cloud) | `text-embedding-3-small` | OpenAI | API |
| Best quality (cloud) | `text-embedding-3-large` | OpenAI | API |

## Database Schema

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    metadata TEXT,
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Error Handling

Vectolite provides clear error messages for common issues:

- **Missing dependencies**: Clear instructions for installing required packages
- **Invalid files**: Validation for file types and existence
- **API errors**: Detailed error messages for embedding service failures
- **Database issues**: Helpful SQLite error handling

## Performance Tips

1. **Use local models** for development and small datasets
2. **Cache models** - SentenceTransformer models are automatically cached
3. **Chunk large documents** for better search granularity
4. **Use appropriate chunk sizes** - 1000-2000 characters work well
5. **Add overlap** between chunks to maintain context

## Troubleshooting

### Common Issues

```bash
# Missing requirements
pip install - requirements.txt

# Permission errors
chmod 755 vectolite.py

# Database locked
# Close other connections to the database file
```

### Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your Vectolite operations
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.