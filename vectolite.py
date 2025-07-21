import sqlite3
import numpy as np
import json
import os
import logging
from typing import List, Optional, Dict, Any, Callable, Tuple
from functools import lru_cache
from pathlib import Path

import typer
from rich import print
from rich.console import Console

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import openai
except ImportError:
    openai = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()
console = Console()


class VectoliteError(Exception):
    """Base exception for Vectolite errors."""
    pass


class EmbeddingError(VectoliteError):
    """Exception raised for embedding-related errors."""
    pass


class Vectolite:
    """A lightweight vector database using SQLite for storage."""
    
    def __init__(
        self, 
        db_path: str = "vectolite.db", 
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None
    ):
        self.db_path = Path(db_path)
        self.embed_fn = embed_fn
        self._ensure_db_directory()
        self._init_db()

    def _ensure_db_directory(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_db(self) -> None:
        """Initialize the database schema."""
        schema = """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            metadata TEXT,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executescript(schema)
                conn.commit()
        except sqlite3.Error as e:
            raise VectoliteError(f"Failed to initialize database: {e}")

    def insert(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Insert a document into the database.
        
        Args:
            text: The text content to embed and store
            metadata: Optional metadata dictionary
            
        Returns:
            The ID of the inserted document
            
        Raises:
            VectoliteError: If no embedding function is provided or insertion fails
        """
        if not self.embed_fn:
            raise VectoliteError("No embedding function provided.")
        
        if not text.strip():
            raise VectoliteError("Text cannot be empty.")

        try:
            embedding = self.embed_fn([text])[0]
            emb_blob = np.array(embedding, dtype='float32').tobytes()
            meta_str = json.dumps(metadata or {})

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO documents (text, metadata, embedding) VALUES (?, ?, ?)",
                    (text, meta_str, emb_blob)
                )
                doc_id = cursor.lastrowid
                conn.commit()
                return doc_id
                
        except Exception as e:
            raise VectoliteError(f"Failed to insert document: {e}")

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query the database for similar documents.
        
        Args:
            query_text: The query text to search for
            top_k: Number of top results to return
            
        Returns:
            List of documents with similarity scores
            
        Raises:
            VectoliteError: If no embedding function is provided or query fails
        """
        if not self.embed_fn:
            raise VectoliteError("No embedding function provided.")
        
        if not query_text.strip():
            raise VectoliteError("Query text cannot be empty.")
            
        if top_k <= 0:
            raise VectoliteError("top_k must be positive.")

        try:
            query_vec = np.array(self.embed_fn([query_text])[0], dtype='float32')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, text, metadata, embedding FROM documents")
                docs = cursor.fetchall()

            if not docs:
                return []

            scored = []
            for doc_id, text, metadata, emb_blob in docs:
                emb = np.frombuffer(emb_blob, dtype='float32')
                
                # Cosine similarity
                dot_product = np.dot(emb, query_vec)
                norm_product = np.linalg.norm(emb) * np.linalg.norm(query_vec)
                
                if norm_product == 0:
                    score = 0.0
                else:
                    score = dot_product / norm_product
                
                scored.append({
                    "id": doc_id,
                    "score": float(score),
                    "text": text,
                    "metadata": json.loads(metadata or "{}")
                })

            return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
            
        except Exception as e:
            raise VectoliteError(f"Failed to query documents: {e}")

    def count_documents(self) -> int:
        """Return the total number of documents in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents")
                return cursor.fetchone()[0]
        except sqlite3.Error as e:
            raise VectoliteError(f"Failed to count documents: {e}")

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document by ID.
        
        Returns:
            True if document was deleted, False if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise VectoliteError(f"Failed to delete document: {e}")

    def list_documents(
        self, 
        limit: int = 50, 
        offset: int = 0,
        include_text: bool = True,
        max_text_length: int = 200
    ) -> List[Dict[str, Any]]:
        """List documents in the database.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            include_text: Whether to include document text
            max_text_length: Maximum length of text to include (truncated with ...)
            
        Returns:
            List of document dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if include_text:
                    cursor.execute(
                        "SELECT id, text, metadata, created_at FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (limit, offset)
                    )
                else:
                    cursor.execute(
                        "SELECT id, metadata, created_at FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (limit, offset)
                    )
                
                docs = cursor.fetchall()
                
                result = []
                for row in docs:
                    if include_text:
                        doc_id, text, metadata, created_at = row
                        # Truncate text if it's too long
                        if len(text) > max_text_length:
                            display_text = text[:max_text_length] + "..."
                        else:
                            display_text = text
                        
                        doc = {
                            "id": doc_id,
                            "text": display_text,
                            "full_text_length": len(text),
                            "metadata": json.loads(metadata or "{}"),
                            "created_at": created_at
                        }
                    else:
                        doc_id, metadata, created_at = row
                        doc = {
                            "id": doc_id,
                            "metadata": json.loads(metadata or "{}"),
                            "created_at": created_at
                        }
                    
                    result.append(doc)
                
                return result
                
        except sqlite3.Error as e:
            raise VectoliteError(f"Failed to list documents: {e}")

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get a single document by ID.
        
        Args:
            doc_id: The document ID
            
        Returns:
            Document dictionary or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, text, metadata, created_at FROM documents WHERE id = ?",
                    (doc_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                doc_id, text, metadata, created_at = row
                return {
                    "id": doc_id,
                    "text": text,
                    "metadata": json.loads(metadata or "{}"),
                    "created_at": created_at
                }
                
        except sqlite3.Error as e:
            raise VectoliteError(f"Failed to get document: {e}")


class EmbeddingProvider:
    """Factory for creating embedding functions."""
    
    @staticmethod
    @lru_cache(maxsize=2)
    def _get_sentence_transformer(model: str) -> 'SentenceTransformer':
        """Cache sentence transformer models."""
        if SentenceTransformer is None:
            raise EmbeddingError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return SentenceTransformer(model)
    
    @staticmethod
    def create_local_embedder(model: str) -> Callable[[List[str]], List[List[float]]]:
        """Create a local embedding function using SentenceTransformers."""
        def local_embed(texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            try:
                model_instance = EmbeddingProvider._get_sentence_transformer(model)
                return model_instance.encode(texts, convert_to_numpy=True).tolist()
            except Exception as e:
                raise EmbeddingError(f"Local embedding failed: {e}")
        
        return local_embed
    
    @staticmethod
    def create_openai_embedder(model: str) -> Callable[[List[str]], List[List[float]]]:
        """Create an OpenAI embedding function."""
        if openai is None:
            raise EmbeddingError("openai not installed. Install with: pip install openai")
        
        try:
            client = openai.OpenAI()
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize OpenAI client: {e}")
        
        def openai_embed(texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            try:
                response = client.embeddings.create(input=texts, model=model)
                return [item.embedding for item in response.data]
            except Exception as e:
                raise EmbeddingError(f"OpenAI embedding failed: {e}")
        
        return openai_embed


def resolve_embed_fn(model: str, local: bool = True) -> Callable[[List[str]], List[List[float]]]:
    """Resolve an embedding function based on parameters.
    
    Args:
        model: The model name/identifier
        local: Whether to use local (True) or OpenAI (False) embeddings
        
    Returns:
        An embedding function
    """
    try:
        if local:
            return EmbeddingProvider.create_local_embedder(model)
        else:
            return EmbeddingProvider.create_openai_embedder(model)
    except Exception as e:
        logger.error(f"Failed to create embedding function: {e}")
        raise


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """Chunk text into smaller segments with optional overlap.
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text.strip():
        return []
    
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph exceeds max_chars, finalize current chunk
        if current_chunk and len(current_chunk) + len(para) + 2 > max_chars:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap if specified
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Add the final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def validate_file_path(file_path: str) -> Path:
    """Validate and return Path object for file."""
    path = Path(file_path)
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {file_path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {file_path}")
    if path.suffix.lower() not in ['.txt', '.md']:
        raise typer.BadParameter(f"Unsupported file type. Use .txt or .md files.")
    return path


@app.command()
def add(
    text: str = typer.Argument(..., help="Text to embed and store"),
    metadata: Optional[str] = typer.Option(None, "--metadata", "-m", help="Optional metadata in JSON format"),
    db: str = typer.Option("vectolite.db", "--db", "-d", help="Database file path"),
    local: bool = typer.Option(True, "--local/--remote", help="Use local embedding model"),
    model: str = typer.Option("all-MiniLM-L6-v2", "--model", help="Embedding model to use")
):
    """Add a document using specified embedding model."""
    try:
        embed_fn = resolve_embed_fn(model, local)
        vdb = Vectolite(db, embed_fn=embed_fn)
        
        meta = json.loads(metadata) if metadata else {}
        doc_id = vdb.insert(text, meta)
        
        print(f"[green]Document added with ID: {doc_id}[/green]")
        
    except json.JSONDecodeError:
        print("[red]Error: Invalid JSON in metadata[/red]")
        raise typer.Exit(1)
    except (VectoliteError, EmbeddingError) as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def query(
    q: str = typer.Argument(..., help="Query text to search for similar documents"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of top results to return"),
    db: str = typer.Option("vectolite.db", "--db", "-d", help="Database file path"),
    local: bool = typer.Option(True, "--local/--remote", help="Use local embedding model"),
    model: str = typer.Option("all-MiniLM-L6-v2", "--model", help="Embedding model to use")
):
    """Query the database using specified embedding model."""
    try:
        embed_fn = resolve_embed_fn(model, local)
        vdb = Vectolite(db, embed_fn=embed_fn)
        results = vdb.query(q, top_k=top_k)

        if not results:
            print("[yellow]No results found.[/yellow]")
            return

        print(f"\n[bold]Found {len(results)} results:[/bold]\n")
        
        for i, res in enumerate(results, 1):
            print(f"[bold]{i}. Score:[/bold] {res['score']:.4f}")
            print(f"[bold]ID:[/bold] {res['id']}")
            print(f"[cyan]Text:[/cyan] {res['text'][:200]}{'...' if len(res['text']) > 200 else ''}")
            if res['metadata']:
                print(f"[yellow]Metadata:[/yellow] {res['metadata']}")
            print()
            
    except (VectoliteError, EmbeddingError) as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def ingest_file(
    file_path: str = typer.Argument(..., help="Path to .txt or .md file"),
    metadata: Optional[str] = typer.Option(None, "--metadata", "-m", help="Optional metadata in JSON format"),
    chunk: bool = typer.Option(True, "--chunk/--no-chunk", help="Chunk the file into smaller parts"),
    max_chars: int = typer.Option(2000, "--max-chars", help="Maximum characters per chunk"),
    overlap: int = typer.Option(200, "--overlap", help="Character overlap between chunks"),
    db: str = typer.Option("vectolite.db", "--db", "-d", help="Database file path"),
    local: bool = typer.Option(True, "--local/--remote", help="Use local embedding model"),
    model: str = typer.Option("all-MiniLM-L6-v2", "--model", help="Embedding model to use")
):
    """Ingest a .txt or .md file into the vector database."""
    try:
        # Validate file
        path = validate_file_path(file_path)
        
        # Read file content
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print("[red]Error: File encoding not supported. Please use UTF-8.[/red]")
            raise typer.Exit(1)
        
        # Parse metadata
        meta = json.loads(metadata) if metadata else {}
        meta.update({"source": str(path), "filename": path.name})
        
        # Initialize database
        embed_fn = resolve_embed_fn(model, local)
        vdb = Vectolite(db, embed_fn=embed_fn)

        # Process text
        if chunk:
            chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
        else:
            chunks = [text]
        
        # Insert chunks
        doc_ids = []
        for i, chunked_text in enumerate(chunks):
            chunk_meta = {**meta, "chunk_index": i, "total_chunks": len(chunks)}
            doc_id = vdb.insert(chunked_text, chunk_meta)
            doc_ids.append(doc_id)

        print(f"[green]Successfully ingested file:[/green] {path}")
        print(f"[green]Created {len(chunks)} chunks with IDs: {doc_ids}[/green]")
        
    except json.JSONDecodeError:
        print("[red]Error: Invalid JSON in metadata[/red]")
        raise typer.Exit(1)
    except (VectoliteError, EmbeddingError) as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stats(
    db: str = typer.Option("vectolite.db", "--db", "-d", help="Database file path")
):
    """Show database statistics."""
    try:
        vdb = Vectolite(db)
        count = vdb.count_documents()
        db_size = Path(db).stat().st_size if Path(db).exists() else 0
        
        print(f"[bold]Database Statistics:[/bold]")
        print(f"  File: {db}")
        print(f"  Documents: {count}")
        print(f"  Size: {db_size / 1024 / 1024:.2f} MB")
        
    except VectoliteError as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def delete(
    doc_id: int = typer.Argument(..., help="Document ID to delete"),
    db: str = typer.Option("vectolite.db", "--db", "-d", help="Database file path")
):
    """Delete a document by ID."""
    try:
        vdb = Vectolite(db)
        if vdb.delete_document(doc_id):
            print(f"[green]Document {doc_id} deleted successfully.[/green]")
        else:
            print(f"[yellow]Document {doc_id} not found.[/yellow]")
            
    except VectoliteError as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list(
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of documents to show"),
    offset: int = typer.Option(0, "--offset", "-o", help="Number of documents to skip"),
    no_text: bool = typer.Option(False, "--no-text", help="Don't show document text"),
    max_text: int = typer.Option(100, "--max-text", help="Maximum text length to display"),
    db: str = typer.Option("vectolite.db", "--db", "-d", help="Database file path")
):
    """List all documents in the database."""
    try:
        vdb = Vectolite(db)
        docs = vdb.list_documents(
            limit=limit, 
            offset=offset, 
            include_text=not no_text,
            max_text_length=max_text
        )
        
        if not docs:
            print("[yellow]No documents found.[/yellow]")
            return
        
        total_count = vdb.count_documents()
        showing_from = offset + 1
        showing_to = min(offset + len(docs), total_count)
        
        print(f"[bold]Showing {showing_from}-{showing_to} of {total_count} documents:[/bold]\n")
        
        for doc in docs:
            print(f"[bold]ID:[/bold] {doc['id']}")
            print(f"[bold]Created:[/bold] {doc['created_at']}")
            
            if not no_text:
                print(f"[cyan]Text:[/cyan] {doc['text']}")
                if doc.get('full_text_length', 0) > max_text:
                    print(f"[dim]  (showing first {max_text} of {doc['full_text_length']} characters)[/dim]")
            
            if doc['metadata']:
                print(f"[yellow]Metadata:[/yellow] {doc['metadata']}")
            
            print()
            
    except VectoliteError as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def show(
    doc_id: int = typer.Argument(..., help="Document ID to show"),
    db: str = typer.Option("vectolite.db", "--db", "-d", help="Database file path")
):
    """Show a specific document by ID."""
    try:
        vdb = Vectolite(db)
        doc = vdb.get_document(doc_id)
        
        if not doc:
            print(f"[yellow]Document {doc_id} not found.[/yellow]")
            return
        
        print(f"[bold]Document ID:[/bold] {doc['id']}")
        print(f"[bold]Created:[/bold] {doc['created_at']}")
        print(f"[bold]Text Length:[/bold] {len(doc['text'])} characters")
        
        if doc['metadata']:
            print(f"[yellow]Metadata:[/yellow] {doc['metadata']}")
        
        print(f"\n[cyan]Full Text:[/cyan]\n{doc['text']}")
            
    except VectoliteError as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()