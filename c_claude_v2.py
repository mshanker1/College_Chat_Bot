import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import os
import json
from datetime import datetime, timedelta
import hashlib
from typing import List, Dict
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Advanced PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- HELPER FUNCTIONS & CLASSES ---

class PDFChatbot:
    """Enhanced PDF chatbot with improved chunking and retrieval strategies."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.pdf_contents: Dict[str, str] = {}
        self.text_chunks: List[Dict] = []
        self.chunk_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.uploads_dir = "uploaded_pdfs"
        self.cache_dir = "pdf_cache"
        self.cache_duration_days = 90
        
        # Initialize directories
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load the sentence transformer model
        @st.cache_resource
        def load_embedding_model(name):
            try:
                if torch.cuda.is_available():
                    device = 'cuda'
                elif torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
                return SentenceTransformer(name, device=device)
            except Exception as e:
                st.error(f"Error loading embedding model '{name}': {e}")
                st.stop()
        
        self.embedding_model = load_embedding_model(model_name)
        self.model_device = self.embedding_model.device

    def get_cache_path(self, identifier: str) -> str:
        """Generates a unique path for cache files based on an identifier."""
        cache_hash = hashlib.md5(identifier.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"cache_{cache_hash}")

    def is_cache_valid(self, cache_path: str) -> bool:
        """Checks if the cache at the given path is still valid."""
        meta_file = f"{cache_path}_meta.json"
        if not os.path.exists(meta_file):
            return False
        
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            cached_at_str = metadata.get('cached_at')
            if not cached_at_str: 
                return False
            
            cached_date = datetime.fromisoformat(cached_at_str)
            return datetime.now() - cached_date < timedelta(days=self.cache_duration_days)
        except Exception as e:
            st.warning(f"Cache metadata corrupted: {e}")
            return False

    def save_to_cache(self, identifier: str) -> None:
        """Saves processed data to cache files."""
        cache_path = self.get_cache_path(identifier)
        
        try:
            metadata = {
                'identifier': identifier,
                'cached_at': datetime.now().isoformat(),
                'files_count': len(self.pdf_contents),
                'chunks_count': len(self.text_chunks)
            }
            with open(f"{cache_path}_meta.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
                
            with open(f"{cache_path}_chunks.json", 'w', encoding='utf-8') as f:
                json.dump(self.text_chunks, f)
                
            if self.chunk_embeddings is not None:
                np.save(f"{cache_path}_embeddings.npy", self.chunk_embeddings.cpu().numpy())
            
            # Save TF-IDF components
            if self.tfidf_vectorizer is not None:
                import pickle
                with open(f"{cache_path}_tfidf_vectorizer.pkl", 'wb') as f:
                    pickle.dump(self.tfidf_vectorizer, f)
                np.save(f"{cache_path}_tfidf_matrix.npy", self.tfidf_matrix.toarray())
                
            st.success(f"Cache saved for identifier: {identifier}")
        except Exception as e:
            st.error(f"Error saving to cache: {e}")

    def load_from_cache(self, identifier: str) -> bool:
        """Loads data from cache if it exists and is valid."""
        cache_path = self.get_cache_path(identifier)
        if not self.is_cache_valid(cache_path):
            return False
            
        try:
            with open(f"{cache_path}_chunks.json", 'r', encoding='utf-8') as f:
                self.text_chunks = json.load(f)
            
            loaded_embeddings = np.load(f"{cache_path}_embeddings.npy")
            self.chunk_embeddings = torch.from_numpy(loaded_embeddings).to(self.model_device)

            # Load TF-IDF components
            import pickle
            try:
                with open(f"{cache_path}_tfidf_vectorizer.pkl", 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                self.tfidf_matrix = np.load(f"{cache_path}_tfidf_matrix.npy")
            except FileNotFoundError:
                # Fallback if TF-IDF cache doesn't exist
                self._create_tfidf_index()

            st.success(f"Cache loaded for identifier: {identifier}")
            return True
        except Exception as e:
            st.warning(f"Could not load from cache: {e}")
            return False

    def get_cache_info(self, identifier: str) -> Dict:
        """Retrieves cache information."""
        cache_path = self.get_cache_path(identifier)
        meta_path = f"{cache_path}_meta.json"
        
        if not os.path.exists(meta_path):
            return {'exists': False}
        
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            cached_at_str = metadata.get('cached_at', '')
            if not cached_at_str:
                return {'exists': True, 'is_valid': False}

            cached_date = datetime.fromisoformat(cached_at_str)
            days_old = (datetime.now() - cached_date).days
            
            return {
                'exists': True,
                'is_valid': days_old < self.cache_duration_days,
                'days_old': days_old,
                **metadata
            }
        except Exception as e:
            return {'exists': False, 'message': f"Error: {e}"}

    def extract_text_from_pdf(self, pdf_path: str, filename: str) -> None:
        """Enhanced PDF text extraction with better cleaning."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Better text cleaning
                # Remove excessive whitespace
                page_text = re.sub(r'\s+', ' ', page_text)
                # Fix hyphenated words across lines
                page_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', page_text)
                # Remove header/footer patterns (basic)
                page_text = re.sub(r'^\d+\s*$', '', page_text, flags=re.MULTILINE)
                
                text += page_text + "\n\n"  # Double newline between pages
            
            doc.close()
            
            # Final cleaning
            text = text.strip()
            # Remove multiple consecutive newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            if len(text) > 100:
                self.pdf_contents[filename] = text
                st.success(f"✅ Extracted {len(text):,} characters from {filename}")
            else:
                st.warning(f"⚠️ Minimal content extracted from {filename}")
                
        except Exception as e:
            st.error(f"❌ Error processing {filename}: {e}")

    def process_uploaded_files(self, uploaded_files) -> None:
        """Process uploaded PDF files."""
        self.pdf_contents = {}
        if not uploaded_files:
            st.warning("No files uploaded.")
            return

        progress_bar = st.progress(0, text="Processing files...")
        
        for i, uploaded_file in enumerate(uploaded_files):
            temp_path = os.path.join(self.uploads_dir, uploaded_file.name)
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                self.extract_text_from_pdf(temp_path, uploaded_file.name)
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files), 
                                text=f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
        
        progress_bar.empty()

    def smart_chunk_text(self, text: str, source: str, 
                        chunk_size: int = 800, overlap: int = 100) -> List[Dict]:
        """
        Improved chunking that preserves semantic boundaries.
        """
        chunks = []
        
        # Split into sentences first
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback if NLTK fails
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                # Add overlap from previous chunk if it exists
                if chunks and overlap > 0:
                    prev_chunk_words = chunks[-1]['text'].split()
                    overlap_words = prev_chunk_words[-min(overlap//5, len(prev_chunk_words)):]
                    chunk_text = ' '.join(overlap_words) + ' ' + chunk_text
                
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'chunk_id': len(chunks)
                })
                
                # Start new chunk with overlap
                if overlap > 0:
                    overlap_sentences = current_chunk[-min(2, len(current_chunk)):]
                    current_chunk = overlap_sentences + [sentence]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunks and overlap > 0:
                prev_chunk_words = chunks[-1]['text'].split()
                overlap_words = prev_chunk_words[-min(overlap//5, len(prev_chunk_words)):]
                chunk_text = ' '.join(overlap_words) + ' ' + chunk_text
            
            chunks.append({
                'text': chunk_text,
                'source': source,
                'chunk_id': len(chunks)
            })
        
        return chunks

    def _create_tfidf_index(self):
        """Create TF-IDF index for keyword-based search."""
        if not self.text_chunks:
            return
            
        chunk_texts = [chunk['text'] for chunk in self.text_chunks]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)

    def create_chunks_and_embeddings(self) -> None:
        """Process PDF content into chunks and generate embeddings."""
        self.text_chunks = []
        if not self.pdf_contents:
            st.error("No PDF content available.")
            return

        # Create chunks
        for filename, content in self.pdf_contents.items():
            file_chunks = self.smart_chunk_text(content, filename)
            self.text_chunks.extend(file_chunks)
        
        if not self.text_chunks:
            st.error("No chunks created from PDFs.")
            return

        st.info(f"Creating embeddings for {len(self.text_chunks)} chunks...")
        
        # Generate semantic embeddings
        chunk_texts = [chunk['text'] for chunk in self.text_chunks]
        try:
            self.chunk_embeddings = self.embedding_model.encode(
                chunk_texts, 
                convert_to_tensor=True, 
                show_progress_bar=True,
                batch_size=32  # Process in batches for better memory usage
            )
            
            # Create TF-IDF index
            self._create_tfidf_index()
            
            st.success("✅ Embeddings and indexes created!")
            
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            self.chunk_embeddings = None

    def hybrid_search(self, question: str, top_k: int = 8) -> List[Dict]:
        """
        Hybrid search combining semantic similarity and keyword matching.
        """
        if self.chunk_embeddings is None or len(self.text_chunks) == 0:
            st.warning("No embeddings available for search.")
            return []
            
        try:
            # Semantic search
            question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
            semantic_scores = util.cos_sim(question_embedding, self.chunk_embeddings)[0]
            
            # Keyword search (TF-IDF)
            keyword_scores = np.zeros(len(self.text_chunks))
            if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
                question_tfidf = self.tfidf_vectorizer.transform([question])
                keyword_similarities = cosine_similarity(question_tfidf, self.tfidf_matrix)[0]
                keyword_scores = keyword_similarities
            
            # Combine scores (weighted average)
            semantic_weight = 0.7
            keyword_weight = 0.3
            
            # Normalize scores to 0-1 range
            semantic_scores_norm = (semantic_scores.cpu().numpy() + 1) / 2  # cos_sim returns -1 to 1
            keyword_scores_norm = keyword_scores
            
            combined_scores = (semantic_weight * semantic_scores_norm + 
                             keyword_weight * keyword_scores_norm)
            
            # Get top results
            top_indices = np.argsort(combined_scores)[-top_k:][::-1]
            
            relevant_chunks = []
            for idx in top_indices:
                chunk = self.text_chunks[idx].copy()
                chunk['semantic_score'] = semantic_scores[idx].item()
                chunk['keyword_score'] = keyword_scores[idx]
                chunk['combined_score'] = combined_scores[idx]
                relevant_chunks.append(chunk)
            
            # Filter out very low scores
            relevant_chunks = [chunk for chunk in relevant_chunks 
                             if chunk['combined_score'] > 0.1]
            
            return relevant_chunks
            
        except Exception as e:
            st.error(f"Error in hybrid search: {e}")
            return []

    def generate_answer(self, question: str, context_chunks: List[Dict], client: OpenAI) -> str:
        """Generate answer with improved prompting."""
        if not context_chunks:
            return "I couldn't find relevant information in the PDF documents to answer your question."
        
        # Prepare context with better formatting
        context_str = ""
        sources = set()
        
        for i, chunk in enumerate(context_chunks):
            context_str += f"=== Context {i+1} (from {chunk['source']}) ===\n"
            context_str += f"{chunk['text']}\n\n"
            sources.add(chunk['source'])
        
        # Enhanced prompt
        prompt = f"""You are an expert document analyst. Answer the user's question based ONLY on the provided context from PDF documents.

CONTEXT FROM DOCUMENTS:
{context_str}

QUESTION: {question}

INSTRUCTIONS:
1. Provide a comprehensive answer using ONLY the information from the context above
2. If the context doesn't contain sufficient information, clearly state: "The provided documents don't contain enough information to fully answer this question."
3. Cite sources by mentioning the document name in brackets, e.g., [document.pdf]
4. If information comes from multiple sources, cite all relevant sources
5. Use clear formatting with paragraphs, bullet points, or lists when helpful
6. Be specific and detailed in your response
7. Do not add information not present in the context
8. If there are multiple perspectives or conflicting information, mention this

Answer:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise document analyst that answers questions based strictly on provided context. Always cite sources and be comprehensive in your responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {e}"


# --- STREAMLIT UI ---

def get_openai_client():
    """Get OpenAI client with API key."""
    api_key = st.secrets.get("OPENAI_API_KEY") 
    
    if not api_key:
        st.warning("🚨 OpenAI API key not found! Please set it in Streamlit secrets.")
        return None
    
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

def main():
    st.title("🤖 Enhanced PDF Chatbot")
    st.markdown("**Improved chunking, hybrid search, and better answer generation**")

    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
        st.session_state.messages = []
        st.session_state.pdfs_processed = False
        st.session_state.last_used_identifier = None

    openai_client = get_openai_client()
    if openai_client is None:
        st.stop()
        
    # Sidebar for PDF management
    with st.sidebar:
        st.header("📄 PDF Management")
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 400, 1200, 800, 50)
            overlap = st.slider("Chunk Overlap", 50, 200, 100, 25)
            search_results = st.slider("Search Results", 3, 15, 8, 1)
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload PDFs for analysis"
        )

        identifier = ""
        if uploaded_files:
            file_details = [f"{f.name}-{f.size}" for f in sorted(uploaded_files, key=lambda x: x.name)]
            identifier = "uploaded_" + hashlib.sha256("|".join(file_details).encode('utf-8')).hexdigest()

        # Cache information
        cache_info_display = {}
        if identifier:
            cache_info_display = st.session_state.chatbot.get_cache_info(identifier)
            if cache_info_display.get('exists'):
                status = "✅ Valid" if cache_info_display['is_valid'] else "⚠️ Expired"
                st.info(f"Cache: {status} ({cache_info_display['days_old']} days old)")

        # Process button
        process_disabled = not uploaded_files or (
            st.session_state.last_used_identifier == identifier and 
            cache_info_display.get('is_valid', False) and 
            st.session_state.pdfs_processed
        )
        
        if process_disabled and st.session_state.pdfs_processed:
            st.success("✅ PDFs processed and ready!")
            
        if st.button("🔄 Process PDFs", disabled=process_disabled):
            st.session_state.pdfs_processed = False
            st.session_state.messages = []
            st.session_state.last_used_identifier = identifier
            
            with st.spinner("Processing documents..."):
                if st.session_state.chatbot.load_from_cache(identifier):
                    st.session_state.pdfs_processed = True
                else:
                    st.session_state.chatbot.process_uploaded_files(uploaded_files)
                    if st.session_state.chatbot.pdf_contents:
                        st.session_state.chatbot.create_chunks_and_embeddings()
                        if st.session_state.chatbot.chunk_embeddings is not None:
                            st.session_state.chatbot.save_to_cache(identifier)
                            st.session_state.pdfs_processed = True
                        else:
                            st.error("Failed to create embeddings.")
                    else:
                        st.error("No content extracted from PDFs.")
            
            if st.session_state.pdfs_processed:
                st.success("✅ PDFs processed successfully!")
                st.rerun()

    # Main chat interface
    if not st.session_state.pdfs_processed:
        st.info("👈 Upload PDFs and click 'Process PDFs' to start chatting.")
        
        # Show sample questions when no PDFs are processed
        st.subheader("💡 Tips for Better Results")
        st.markdown("""
        **This enhanced version includes:**
        - **Smarter chunking** that preserves sentence boundaries
        - **Hybrid search** combining semantic similarity and keyword matching
        - **Better text cleaning** to remove noise from PDFs
        - **Improved prompting** for more accurate answers
        - **Enhanced caching** for faster repeated queries
        
        **For best results:**
        - Ask specific, detailed questions
        - Use keywords that might appear in the documents
        - Break complex questions into simpler parts
        """)
    else:
        st.header("💬 Chat with your PDFs")
        
        # Display performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", len(st.session_state.chatbot.pdf_contents))
        with col2:
            st.metric("Text Chunks", len(st.session_state.chatbot.text_chunks))
        with col3:
            st.metric("Chat Messages", len(st.session_state.messages))
        
        # Settings toggle
        with st.sidebar:
            st.markdown("---")
            show_context = st.checkbox("Show Retrieved Context", value=False)
            show_scores = st.checkbox("Show Relevance Scores", value=False)
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "context" in message and show_context:
                    with st.expander("📄 Retrieved Context"):
                        if message["context"]:
                            for i, chunk in enumerate(message["context"]):
                                score_info = ""
                                if show_scores:
                                    score_info = f" (Score: {chunk.get('combined_score', 0):.3f})"
                                
                                st.info(f"**Chunk {i+1}** from {chunk.get('source', 'Unknown')}{score_info}")
                                st.text(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                        else:
                            st.warning("No relevant context found.")

        # Chat input
        if question := st.chat_input("Ask about your PDFs..."):
            st.session_state.messages.append({"role": "user", "content": question})
            
            with st.chat_message("user"):
                st.markdown(question)
                
            with st.chat_message("assistant"):
                with st.spinner("Searching and generating answer..."):
                    # Use hybrid search with user-configured parameters
                    relevant_chunks = st.session_state.chatbot.hybrid_search(
                        question, top_k=search_results
                    )
                    
                    answer = st.session_state.chatbot.generate_answer(
                        question, relevant_chunks, openai_client
                    )
                    
                    st.markdown(answer)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "context": relevant_chunks
            })
            
    # Sidebar controls
    if st.session_state.messages:
        with st.sidebar:
            st.markdown("---")
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
                
            # Export chat option
            if st.button("📥 Export Chat"):
                chat_export = []
                for msg in st.session_state.messages:
                    chat_export.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": datetime.now().isoformat()
                    })
                
                st.download_button(
                    "Download Chat History",
                    data=json.dumps(chat_export, indent=2),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()