# vectolite_ui.py
import streamlit as st
from vectolite import Vectolite, resolve_embed_fn, chunk_text, VectoliteError, EmbeddingError
import json
import os
from pathlib import Path
import traceback
import time

st.set_page_config(page_title="Vectolite UI", layout="wide", page_icon="🧠")

# Custom CSS for better styling
st.markdown("""
<style>
    .result-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .score-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vdb' not in st.session_state:
    st.session_state.vdb = None
if 'current_config' not in st.session_state:
    st.session_state.current_config = {}

# Sidebar configuration
st.sidebar.title("🔍 Vectolite Configuration")

with st.sidebar:
    st.markdown("### Database Settings")
    db_path = st.text_input("📦 Database Path", value="vectolite.db", help="Path to your SQLite database file")
    
    st.markdown("### Model Settings")
    local = st.checkbox("🏠 Use local model", value=True, help="Use HuggingFace models locally vs OpenAI API")
    
    if local:
        model_options = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2", 
            "paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/all-MiniLM-L6-v2"
        ]
        model = st.selectbox("🤖 Local Model", options=model_options, help="HuggingFace model for embeddings")
    else:
        openai_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002"
        ]
        model = st.selectbox("🌐 OpenAI Model", options=openai_models, help="OpenAI embedding model")
        
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("⚠️ OPENAI_API_KEY environment variable not found!")
    
    # Check if configuration changed
    current_config = {"db_path": db_path, "local": local, "model": model}
    config_changed = st.session_state.current_config != current_config
    
    if config_changed or st.button("🔄 Reload Database"):
        try:
            with st.spinner("Loading embedding model..."):
                embed_fn = resolve_embed_fn(model, local=local)
                st.session_state.vdb = Vectolite(db_path=db_path, embed_fn=embed_fn)
                st.session_state.current_config = current_config
            st.success("✅ Database loaded successfully!")
        except (VectoliteError, EmbeddingError) as e:
            st.error(f"❌ Configuration Error: {str(e)}")
            st.session_state.vdb = None
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")
            st.session_state.vdb = None

# Main content
st.title("Vectolite: Vector Search Playground")

# Check if database is loaded
if st.session_state.vdb is None:
    st.warning("⚠️ Please configure and load a database in the sidebar first.")
    st.stop()

vdb = st.session_state.vdb

# Display database stats
try:
    doc_count = vdb.count_documents()
    db_size = Path(db_path).stat().st_size if Path(db_path).exists() else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Documents", doc_count)
    with col2:
        st.metric("💾 Database Size", f"{db_size / 1024 / 1024:.2f} MB")
    with col3:
        st.metric("🤖 Model Type", "Local" if local else "OpenAI")
except Exception as e:
    st.error(f"Error loading database stats: {str(e)}")

# Tabs for different operations
tab1, tab2, tab3, tab4 = st.tabs(["📤 Add Documents", "🔎 Search", "📋 Browse", "⚙️ Manage"])

# Tab 1: Add Documents
with tab1:
    st.subheader("📤 Add Documents to Database")
    
    # Method selection
    add_method = st.radio("Choose input method:", ["📝 Text Input", "📁 File Upload"], horizontal=True)
    
    if add_method == "📝 Text Input":
        st.markdown("### Direct Text Input")
        text_input = st.text_area("Enter text to add:", height=150, placeholder="Type or paste your text here...")
        
        # Metadata input
        with st.expander("🏷️ Optional Metadata"):
            metadata_json = st.text_area(
                "Metadata (JSON format):", 
                value='{"source": "manual_input"}',
                height=100,
                help="Enter metadata as valid JSON"
            )
        
        if st.button("➕ Add Text", type="primary") and text_input.strip():
            try:
                metadata = json.loads(metadata_json) if metadata_json.strip() else {}
                
                # Add progress bar for insert operation
                progress_placeholder = st.empty()
                with progress_placeholder:
                    with st.spinner("🔄 Generating embeddings and inserting document..."):
                        start_time = time.time()
                        doc_id = vdb.insert(text_input.strip(), metadata)
                        elapsed_time = time.time() - start_time
                
                progress_placeholder.empty()
                st.success(f"✅ Text added successfully! Document ID: {doc_id} (took {elapsed_time:.2f}s)")
                st.rerun()
                
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON in metadata field")
            except (VectoliteError, EmbeddingError) as e:
                st.error(f"❌ Error adding document: {str(e)}")
    
    else:  # File Upload
        st.markdown("### File Upload")
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=["txt", "md"], 
            help="Upload .txt or .md files"
        )
        
        if uploaded_file:
            try:
                content = uploaded_file.read().decode("utf-8")
                st.text_area("File Preview:", value=content[:500] + "..." if len(content) > 500 else content, height=150, disabled=True)
                
                # Chunking options
                col1, col2 = st.columns(2)
                with col1:
                    do_chunk = st.checkbox("🔪 Enable chunking", value=True, help="Split large files into smaller chunks")
                with col2:
                    if do_chunk:
                        max_chars = st.number_input("Max characters per chunk", min_value=500, max_value=5000, value=2000)
                        overlap = st.number_input("Overlap between chunks", min_value=0, max_value=500, value=200)
                
                # Metadata
                with st.expander("🏷️ File Metadata"):
                    file_metadata = st.text_area(
                        "Additional metadata (JSON):", 
                        value=f'{{"filename": "{uploaded_file.name}", "file_type": "{uploaded_file.type}"}}',
                        height=80
                    )
                
                if st.button("📁 Ingest File", type="primary"):
                    try:
                        metadata = json.loads(file_metadata) if file_metadata.strip() else {}
                        
                        if do_chunk:
                            chunks = chunk_text(content, max_chars=max_chars, overlap=overlap)
                        else:
                            chunks = [content]
                        
                        # Enhanced progress tracking for file ingestion
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        doc_ids = []
                        start_time = time.time()
                        
                        for i, chunk in enumerate(chunks):
                            chunk_meta = {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
                            
                            # Update status for each chunk
                            status_text.text(f"🔄 Processing chunk {i+1}/{len(chunks)} - Generating embeddings...")
                            
                            with st.spinner(f"Processing chunk {i+1}/{len(chunks)}..."):
                                doc_id = vdb.insert(chunk, chunk_meta)
                                doc_ids.append(doc_id)
                            
                            progress_bar.progress((i + 1) / len(chunks))
                        
                        elapsed_time = time.time() - start_time
                        status_text.empty()
                        
                        st.success(f"✅ File ingested successfully! Created {len(chunks)} chunks with IDs: {doc_ids} (took {elapsed_time:.2f}s)")
                        st.rerun()
                        
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON in metadata field")
                    except (VectoliteError, EmbeddingError) as e:
                        st.error(f"❌ Error ingesting file: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
                        
            except UnicodeDecodeError:
                st.error("❌ File encoding not supported. Please use UTF-8 encoded files.")

# Tab 2: Search
with tab2:
    st.subheader("🔎 Search Documents")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("🔍 Enter search query:", placeholder="What are you looking for?")
    with col2:
        top_k = st.slider("📊 Results", min_value=1, max_value=20, value=5)
    
    if st.button("🚀 Search", type="primary") and query.strip():
        try:
            # Enhanced progress indication for search
            progress_placeholder = st.empty()
            with progress_placeholder:
                with st.spinner("🔍 Generating query embeddings and searching..."):
                    start_time = time.time()
                    results = vdb.query(query.strip(), top_k=top_k)
                    elapsed_time = time.time() - start_time
            
            progress_placeholder.empty()
            
            if not results:
                st.info("🤷 No results found. Try a different query.")
            else:
                st.success(f"✅ Found {len(results)} results in {elapsed_time:.3f}s")
                
                for i, res in enumerate(results, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="result-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <h4>🔹 Result {i}</h4>
                                <span class="score-badge">Score: {res['score']:.4f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"**Document ID:** {res.get('id', 'N/A')}")
                        
                        # Text with expandable view
                        text_preview = res['text'][:300] + "..." if len(res['text']) > 300 else res['text']
                        st.markdown(f"**Text Preview:** {text_preview}")
                        
                        if len(res['text']) > 300:
                            with st.expander("📖 View full text"):
                                st.write(res['text'])
                        
                        # Metadata
                        if res['metadata']:
                            with st.expander("🏷️ View metadata"):
                                st.json(res['metadata'])
                        
                        st.markdown("---")
                        
        except (VectoliteError, EmbeddingError) as e:
            st.error(f"❌ Search Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Tab 3: Browse Documents
with tab3:
    st.subheader("📋 Browse All Documents")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        limit = st.selectbox("Documents per page", options=[5, 10, 20, 50], index=1)
    with col2:
        offset = st.number_input("Skip documents", min_value=0, value=0, step=limit)
    with col3:
        show_text = st.checkbox("Show text content", value=True)
    
    try:
        docs = vdb.list_documents(limit=limit, offset=offset, include_text=show_text, max_text_length=200)
        total_docs = vdb.count_documents()
        
        if docs:
            st.info(f"📄 Showing documents {offset + 1}-{min(offset + len(docs), total_docs)} of {total_docs}")
            
            for doc in docs:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**ID:** {doc['id']} | **Created:** {doc['created_at']}")
                        
                        if show_text:
                            st.write(f"**Text:** {doc['text']}")
                            if doc.get('full_text_length', 0) > 200:
                                st.caption(f"Showing first 200 of {doc['full_text_length']} characters")
                        
                        if doc['metadata']:
                            with st.expander("🏷️ Metadata"):
                                st.json(doc['metadata'])
                    
                    with col2:
                        if st.button(f"🗑️ Delete", key=f"delete_{doc['id']}", help="Delete this document"):
                            with st.spinner(f"🗑️ Deleting document {doc['id']}..."):
                                if vdb.delete_document(doc['id']):
                                    st.success(f"✅ Document {doc['id']} deleted")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to delete document {doc['id']}")
                
                st.markdown("---")
        else:
            st.info("📭 No documents found in the specified range.")
            
    except (VectoliteError, EmbeddingError) as e:
        st.error(f"❌ Error browsing documents: {str(e)}")

# Tab 4: Database Management
with tab4:
    st.subheader("⚙️ Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Database Information")
        try:
            stats_info = {
                "Database Path": db_path,
                "Model Type": "Local HuggingFace" if local else "OpenAI API",
                "Model Name": model,
                "Total Documents": vdb.count_documents(),
                "Database Size": f"{Path(db_path).stat().st_size / 1024 / 1024:.2f} MB" if Path(db_path).exists() else "0 MB"
            }
            
            for key, value in stats_info.items():
                st.metric(key, value)
                
        except Exception as e:
            st.error(f"Error loading database info: {str(e)}")
    
    with col2:
        st.markdown("### 🛠️ Database Operations")
        
        if st.button("🔄 Refresh Stats", help="Reload database statistics"):
            st.rerun()
        
        st.markdown("### 🗑️ Danger Zone")
        with st.expander("⚠️ Advanced Operations", expanded=False):
            st.warning("These operations cannot be undone!")
            
            doc_id_to_delete = st.number_input("Document ID to delete:", min_value=1, step=1)
            if st.button("🗑️ Delete Document by ID", type="secondary"):
                try:
                    with st.spinner(f"🗑️ Deleting document {doc_id_to_delete}..."):
                        if vdb.delete_document(doc_id_to_delete):
                            st.success(f"✅ Document {doc_id_to_delete} deleted successfully")
                            st.rerun()
                        else:
                            st.error(f"❌ Document {doc_id_to_delete} not found")
                except Exception as e:
                    st.error(f"❌ Error deleting document: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Vectolite")