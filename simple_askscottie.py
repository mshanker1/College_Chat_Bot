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
from typing import List, Dict, Optional
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import warnings
import time
import requests
import glob
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Ask Scottie - Maryville College Academic Assistant",
    page_icon="🐕‍🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple styling that definitely works
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #5B0F1B, #EC5E1A);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .info-box {
        background: #E3F2FD;
        border-left: 4px solid #1976D2;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .source-item {
        background: #F5F5F5;
        border: 1px solid #DDD;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- INSTITUTIONAL CONFIGURATION ---
class Config:
    """Configuration for institutional deployment."""
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    SEARCH_RESULTS = 13
    MODEL_NAME = 'all-MiniLM-L6-v2'
    CACHE_DURATION_DAYS = 90
    BATCH_SIZE = 32
    MAX_RETRIES = 2
    
    # Institutional PDF directory and web sources
    INSTITUTIONAL_PDF_DIR = "institutional_pdfs"
    INSTITUTIONAL_WEB_SOURCES = "institutional_web_sources.json"
    CACHE_DIR = "institutional_cache"
    
    # Context-aware settings
    MAX_CONVERSATION_TOKENS = 2000
    CONTEXT_WINDOW_SIZE = 10

class InstitutionalPDFChatbot:
    """PDF chatbot for institutional deployment."""
    
    def __init__(self):
        self.pdf_contents: Dict[str, str] = {}
        self.web_contents: Dict[str, str] = {}
        self.text_chunks: List[Dict] = []
        self.chunk_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Initialize directories
        os.makedirs(Config.INSTITUTIONAL_PDF_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        
        # Load the sentence transformer model
        self.embedding_model = self._load_embedding_model()
        if self.embedding_model:
            self.model_device = self.embedding_model.device
        else:
            st.error("Could not load embedding model.")
            st.stop()

    @st.cache_resource
    def _load_embedding_model(_self):
        """Load embedding model."""
        try:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
            
            cache_folder = os.path.join(os.getcwd(), "model_cache")
            os.makedirs(cache_folder, exist_ok=True)
            
            model = SentenceTransformer(
                Config.MODEL_NAME, 
                device=device,
                cache_folder=cache_folder
            )
            
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    def get_cache_path(self, identifier: str) -> str:
        """Generates cache path for institutional documents."""
        cache_hash = hashlib.md5(identifier.encode('utf-8')).hexdigest()
        return os.path.join(Config.CACHE_DIR, f"institutional_{cache_hash}")

    def is_cache_valid(self, cache_path: str) -> bool:
        """Check if institutional cache is valid."""
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
            return datetime.now() - cached_date < timedelta(days=Config.CACHE_DURATION_DAYS)
        except:
            return False

    def save_to_cache(self, identifier: str) -> None:
        """Save processed institutional data to cache."""
        cache_path = self.get_cache_path(identifier)
        
        try:
            metadata = {
                'identifier': identifier,
                'cached_at': datetime.now().isoformat(),
                'files_count': len(self.pdf_contents) + len(self.web_contents),
                'chunks_count': len(self.text_chunks),
                'config': {
                    'chunk_size': Config.CHUNK_SIZE,
                    'chunk_overlap': Config.CHUNK_OVERLAP,
                    'search_results': Config.SEARCH_RESULTS
                }
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
                
        except Exception as e:
            st.error(f"Error saving to cache: {e}")

    def load_from_cache(self, identifier: str) -> bool:
        """Load institutional data from cache."""
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
                self._create_tfidf_index()

            return True
        except Exception as e:
            return False

    def extract_text_from_pdf(self, pdf_path: str, filename: str) -> None:
        """Extract text from PDF."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                page_text = re.sub(r'\s+', ' ', page_text)
                page_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', page_text)
                text += page_text + "\n\n"
            
            doc.close()
            
            text = text.strip()
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            if len(text) > 100:
                self.pdf_contents[filename] = text
                st.success(f"✅ Extracted {len(text):,} characters from {filename}")
            else:
                st.warning(f"⚠️ Minimal content extracted from {filename}")
            
        except Exception as e:
            st.error(f"❌ Error processing {filename}: {e}")

    def extract_text_from_webpage(self, url: str) -> None:
        """Extract text content from a webpage."""
        try:
            headers = {
                'User-Agent': 'Ask Scottie Academic Chatbot (Educational Use)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text from main content areas
            main_selectors = ['main', 'article', '.content', '.main-content', '#content', '#main']
            main_content = None
            
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if main_content:
                text_content = main_content.get_text()
            else:
                body = soup.find('body')
                if body:
                    text_content = body.get_text()
                else:
                    text_content = soup.get_text()
            
            # Clean the text
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = ' '.join(chunk for chunk in chunks if chunk)
            
            text_content = re.sub(r'\s+', ' ', text_content)
            
            if len(text_content) > 200:
                domain = urlparse(url).netloc
                page_title = soup.find('title')
                if page_title:
                    identifier = f"{domain} - {page_title.get_text().strip()}"
                else:
                    identifier = domain
                
                self.web_contents[identifier] = text_content
                st.success(f"✅ Extracted {len(text_content):,} characters from {identifier}")
            else:
                st.warning(f"⚠️ Minimal content extracted from {url}")
                
        except Exception as e:
            st.error(f"❌ Error processing {url}: {e}")

    def load_web_sources(self) -> bool:
        """Load web pages from the configured sources file."""
        web_sources_file = Config.INSTITUTIONAL_WEB_SOURCES
        
        if not os.path.exists(web_sources_file):
            return False
        
        try:
            with open(web_sources_file, 'r', encoding='utf-8') as f:
                web_sources = json.load(f)
            
            if not web_sources or 'urls' not in web_sources:
                return False
            
            urls = web_sources['urls']
            if not urls:
                return False
            
            progress_bar = st.progress(0, text="Processing web sources...")
            
            for i, url in enumerate(urls):
                st.info(f"Processing: {url}")
                self.extract_text_from_webpage(url)
                
                progress_bar.progress((i + 1) / len(urls), 
                                    text=f"Processing web source ({i+1}/{len(urls)})")
            
            progress_bar.empty()
            return len(self.web_contents) > 0
            
        except Exception as e:
            st.error(f"Error loading web sources: {e}")
            return False

    def load_institutional_pdfs(self) -> bool:
        """Load all PDFs from the institutional directory."""
        self.pdf_contents = {}
        
        pdf_files = glob.glob(os.path.join(Config.INSTITUTIONAL_PDF_DIR, "*.pdf"))
        
        if not pdf_files:
            return False
        
        progress_bar = st.progress(0, text="Processing PDF files...")
        
        for i, pdf_path in enumerate(pdf_files):
            filename = os.path.basename(pdf_path)
            self.extract_text_from_pdf(pdf_path, filename)
            
            progress_bar.progress((i + 1) / len(pdf_files), 
                                text=f"Processing {filename} ({i+1}/{len(pdf_files)})")
        
        progress_bar.empty()
        
        return len(self.pdf_contents) > 0

    def load_all_sources(self) -> bool:
        """Load both PDFs and web sources."""
        pdf_success = self.load_institutional_pdfs()
        web_success = self.load_web_sources()
        
        return pdf_success or web_success

    def smart_chunk_text(self, text: str, source: str) -> List[Dict]:
        """Enhanced chunking that preserves academic requirement structures."""
        chunks = []
        
        # Academic requirement patterns
        requirement_patterns = [
            r'General Education Requirements',
            r'Category [IVX]+ Domains?',
            r'Foundations?\s*\(\d+\s*Credit Hours?\)',
            r'Required courses? include:?',
            r'Completion of one',
            r'The Maryville Curriculum',
            r'Undergraduate Degree Requirements'
        ]
        
        is_requirements_section = any(re.search(pattern, text, re.IGNORECASE) 
                                     for pattern in requirement_patterns)
        
        if is_requirements_section:
            chunk_size = 1400
            overlap_size = 300
        else:
            chunk_size = Config.CHUNK_SIZE
            overlap_size = Config.CHUNK_OVERLAP
        
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                if chunks and overlap_size > 0:
                    prev_chunk_words = chunks[-1]['text'].split()
                    overlap_words = prev_chunk_words[-min(overlap_size//5, len(prev_chunk_words)):]
                    chunk_text = ' '.join(overlap_words) + ' ' + chunk_text
                
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'chunk_id': len(chunks),
                    'is_requirements': is_requirements_section
                })
                
                if overlap_size > 0:
                    overlap_sentences = current_chunk[-min(3, len(current_chunk)):]
                    current_chunk = overlap_sentences + [sentence]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunks and overlap_size > 0:
                prev_chunk_words = chunks[-1]['text'].split()
                overlap_words = prev_chunk_words[-min(overlap_size//5, len(prev_chunk_words)):]
                chunk_text = ' '.join(overlap_words) + ' ' + chunk_text
            
            chunks.append({
                'text': chunk_text,
                'source': source,
                'chunk_id': len(chunks),
                'is_requirements': is_requirements_section
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
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)

    def create_chunks_and_embeddings(self) -> bool:
        """Process PDF and web content into chunks and generate embeddings."""
        self.text_chunks = []
        if not self.pdf_contents and not self.web_contents:
            st.error("No PDF or web content available.")
            return False

        # Create chunks from PDFs
        for filename, content in self.pdf_contents.items():
            file_chunks = self.smart_chunk_text(content, f"📄 {filename}")
            self.text_chunks.extend(file_chunks)
        
        # Create chunks from web sources
        for source_name, content in self.web_contents.items():
            web_chunks = self.smart_chunk_text(content, f"🌐 {source_name}")
            self.text_chunks.extend(web_chunks)
        
        if not self.text_chunks:
            st.error("No chunks created from sources.")
            return False

        st.info(f"Creating embeddings for {len(self.text_chunks)} chunks...")

        chunk_texts = [chunk['text'] for chunk in self.text_chunks]
        try:
            self.chunk_embeddings = self.embedding_model.encode(
                chunk_texts, 
                convert_to_tensor=True, 
                show_progress_bar=True,
                batch_size=32
            )
            
            self._create_tfidf_index()
            
            st.success("✅ Embeddings and indexes created!")
            return True
            
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            self.chunk_embeddings = None
            return False

    def context_aware_search(self, question: str, conversation_history: List[Dict] = None) -> List[Dict]:
        """Enhanced search for comprehensive requirement queries."""
        if self.chunk_embeddings is None or len(self.text_chunks) == 0:
            st.warning("No embeddings available for search.")
            return []
        
        comprehensive_keywords = [
            'general education requirements', 'curriculum requirements', 'degree requirements',
            'all requirements', 'what do i need to take', 'complete requirements',
            'graduation requirements', 'core curriculum'
        ]
        
        is_comprehensive_query = any(keyword in question.lower() for keyword in comprehensive_keywords)
        
        enhanced_question = question
        if is_comprehensive_query:
            enhanced_question = f"{question} foundations category domains credit hours courses curriculum general education requirements"
        
        try:
            question_embedding = self.embedding_model.encode(enhanced_question, convert_to_tensor=True)
            semantic_scores = util.cos_sim(question_embedding, self.chunk_embeddings)[0]
            
            keyword_scores = np.zeros(len(self.text_chunks))
            if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
                question_tfidf = self.tfidf_vectorizer.transform([enhanced_question])
                keyword_similarities = cosine_similarity(question_tfidf, self.tfidf_matrix)[0]
                keyword_scores = keyword_similarities
            
            if is_comprehensive_query:
                for i, chunk in enumerate(self.text_chunks):
                    if chunk.get('is_requirements', False):
                        semantic_scores[i] = semantic_scores[i] * 1.5
            
            semantic_weight = 0.7
            keyword_weight = 0.3
            
            semantic_scores_norm = (semantic_scores.cpu().numpy() + 1) / 2
            keyword_scores_norm = keyword_scores
            
            combined_scores = (semantic_weight * semantic_scores_norm + 
                             keyword_weight * keyword_scores_norm)
            
            if is_comprehensive_query:
                num_results = min(25, len(self.text_chunks))
            else:
                num_results = Config.SEARCH_RESULTS
            
            top_indices = np.argsort(combined_scores)[-num_results:][::-1]
            
            relevant_chunks = []
            for idx in top_indices:
                chunk = self.text_chunks[idx].copy()
                chunk['semantic_score'] = semantic_scores[idx].item()
                chunk['keyword_score'] = keyword_scores[idx]
                chunk['combined_score'] = combined_scores[idx]
                relevant_chunks.append(chunk)
            
            min_score = 0.05 if is_comprehensive_query else 0.1
            relevant_chunks = [chunk for chunk in relevant_chunks 
                             if chunk['combined_score'] > min_score]
            
            return relevant_chunks
            
        except Exception as e:
            st.error(f"Error in search: {e}")
            return []

    def generate_answer(self, question: str, context_chunks: List[Dict], client: OpenAI, 
                       conversation_history: List[Dict] = None) -> str:
        """Generate answer with enhanced requirement handling."""
        if not context_chunks:
            return "I couldn't find relevant information in the documents to answer your question."
        
        comprehensive_keywords = [
            'general education requirements', 'curriculum requirements', 'degree requirements',
            'all requirements', 'what do i need', 'graduation requirements'
        ]
        is_comprehensive_query = any(keyword in question.lower() for keyword in comprehensive_keywords)
        
        context_str = ""
        for i, chunk in enumerate(context_chunks):
            context_str += f"=== Context {i+1} (from {chunk['source']}) ===\n"
            context_str += f"{chunk['text']}\n\n"
        
        base_prompt = f"""You are 'Ask Scottie', Maryville College's helpful academic assistant. You ONLY answer questions about information found in the Maryville College Academic Catalog.

CURRENT QUESTION: {question}

CONTEXT FROM DOCUMENTS:
{context_str}

INSTRUCTIONS:
1. Answer the current question based on the provided document context
2. If the context doesn't contain sufficient information, clearly state what's missing
3. Cite sources by mentioning the document name in brackets, e.g., [document.pdf]
4. Be specific and detailed in your response"""

        if is_comprehensive_query:
            comprehensive_instruction = """

SPECIAL INSTRUCTIONS FOR COMPREHENSIVE REQUIREMENTS:
This question asks for comprehensive general education requirements. Please provide ALL categories and requirements found in the context, including:
- Foundations requirements (with credit hours)
- All Category I, II, III, and IV domains (with credit hours for each)
- Specific course options within each category
- Any special conditions, notes, or placement exam exemptions

Structure your response clearly with headers for each major category and list all course options and credit hour requirements."""
            
            base_prompt += comprehensive_instruction

        prompt = base_prompt + "\n\nAnswer:"

        try:
            max_tokens = 3000 if is_comprehensive_query else 2000
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {e}"

def get_openai_client():
    """Get OpenAI client with API key."""
    api_key = st.secrets.get("OPENAI_API_KEY") 
    
    if not api_key:
        st.warning("🔑 OpenAI API key not found in Streamlit secrets.")
        
        with st.expander("🔧 API Key Setup Instructions", expanded=True):
            st.markdown("""
            **To set up your OpenAI API key:**
            
            1. **For local development:** Create a `.streamlit/secrets.toml` file:
               ```toml
               OPENAI_API_KEY = "your-api-key-here"
               ```
            
            2. **For Streamlit Cloud:** Add the key in your app's secrets section
            
            **Get your API key from:** https://platform.openai.com/api-keys
            """)
        
        return None
    
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI: {e}")
        return None

def initialize_chatbot():
    """Initialize chatbot with institutional documents."""
    if 'chatbot_initialized' not in st.session_state:
        chatbot = InstitutionalPDFChatbot()
        
        pdf_files = glob.glob(os.path.join(Config.INSTITUTIONAL_PDF_DIR, "*.pdf"))
        web_sources_file = Config.INSTITUTIONAL_WEB_SOURCES
        
        if not pdf_files and not os.path.exists(web_sources_file):
            st.warning(f"⚠️ No PDF files or web sources found.")
            
            with st.expander("📋 Setup Instructions", expanded=True):
                st.markdown("""
                **To set up Ask Scottie's Academic Catalog knowledge base:**
                
                **Option 1: PDF Files**
                1. Create a folder named `institutional_pdfs` in the same directory as this app
                2. Add your **Maryville College Academic Catalog PDF documents** to this folder
                
                **Option 2: Web Sources**
                1. Create a file named `institutional_web_sources.json` in the same directory as this app
                2. Add URLs to scrape in this format:
                ```json
                {
                  "urls": [
                    "https://www.maryvillecollege.edu/academics/catalog/",
                    "https://www.maryvillecollege.edu/admissions/requirements/"
                  ]
                }
                ```
                """)
            
            os.makedirs(Config.INSTITUTIONAL_PDF_DIR, exist_ok=True)
            
            st.session_state.chatbot = None
            st.session_state.chatbot_initialized = False
            return None
        
        with st.spinner("Initializing knowledge base..."):
            file_stats = []
            
            for pdf_path in sorted(pdf_files):
                stat = os.stat(pdf_path)
                file_stats.append(f"pdf-{os.path.basename(pdf_path)}-{stat.st_size}-{stat.st_mtime}")
            
            if os.path.exists(web_sources_file):
                stat = os.stat(web_sources_file)
                file_stats.append(f"web-sources-{stat.st_size}-{stat.st_mtime}")
            
            identifier = "institutional_" + hashlib.sha256("|".join(file_stats).encode('utf-8')).hexdigest()
            
            if chatbot.load_from_cache(identifier):
                st.success("✅ Knowledge base loaded from cache")
            else:
                if chatbot.load_all_sources():
                    success = chatbot.create_chunks_and_embeddings()
                    if success:
                        chatbot.save_to_cache(identifier)
                        st.success("✅ Knowledge base initialized and cached")
                    else:
                        st.error("Failed to create embeddings")
                        return None
                else:
                    st.error("Failed to load any sources")
                    return None
            
            st.session_state.chatbot = chatbot
            st.session_state.chatbot_initialized = True
            st.session_state.messages = []
            
    return st.session_state.chatbot

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🐕‍🦺 Ask Scottie</h1>
        <p>Scottish Terrier Academic Assistant • Maryville College Academic Catalog Helper</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Scope limitation notice
    st.markdown("""
    <div class="info-box">
        <strong>📚 Ask Scottie can only answer questions about the Maryville College Academic Catalog.</strong><br>
        For other inquiries, please contact the appropriate college department.
    </div>
    """, unsafe_allow_html=True)

    # Initialize OpenAI client
    openai_client = get_openai_client()
    if openai_client is None:
        st.stop()
    
    # Initialize chatbot
    chatbot = initialize_chatbot()
    if chatbot is None:
        st.info("Please add Academic Catalog sources to get started.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("✅ App Ready")
        with col2:
            st.success("✅ OpenAI OK")
        with col3:
            st.error("❌ No Sources")
        
        if st.button("🔄 Refresh After Adding Sources"):
            st.rerun()
        
        return
    
    # Display system info in sidebar
    with st.sidebar:
        st.header("📊 System Information")
        st.metric("PDF Documents", len(chatbot.pdf_contents))
        st.metric("Web Sources", len(chatbot.web_contents))
        st.metric("Knowledge Chunks", len(chatbot.text_chunks))
        st.metric("Chat Messages", len(st.session_state.messages))
        
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

    # Display initial content if no messages
    if st.session_state.messages == []:
        # Available sources
        if chatbot.pdf_contents or chatbot.web_contents:
            st.subheader("📚 Available Sources")
            
            # Show PDF documents
            if chatbot.pdf_contents:
                st.markdown("**📄 PDF Documents:**")
                for doc_name in chatbot.pdf_contents.keys():
                    st.markdown(f'<div class="source-item">📄 {doc_name}</div>', unsafe_allow_html=True)
            
            # Show web sources
            if chatbot.web_contents:
                st.markdown("**🌐 Web Sources:**")
                for web_name in chatbot.web_contents.keys():
                    st.markdown(f'<div class="source-item">🌐 {web_name}</div>', unsafe_allow_html=True)
        
        # Sample questions
        st.subheader("💡 Try These Sample Questions")
        
        sample_questions = [
            "What are the general education requirements?",
            "What is the grading policy?",
            "Tell me about admission requirements",
            "What are the graduation requirements?",
            "Tell me about the Maryville Curriculum"
        ]
        
        # Two columns for sample questions
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(question, key=f"sample_{i}"):
                    st.session_state.messages.append({"role": "user", "content": question})
                    
                    with st.spinner("🔍 Searching the Academic Catalog..."):
                        relevant_chunks = chatbot.context_aware_search(
                            question, 
                            conversation_history=st.session_state.messages
                        )
                        
                        answer = chatbot.generate_answer(
                            question, 
                            relevant_chunks, 
                            openai_client,
                            conversation_history=st.session_state.messages
                        )
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "system":
            st.markdown(f'<div style="text-align: center; color: #666; margin: 10px 0; font-style: italic;">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask Scottie about the Academic Catalog..."):
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching the Academic Catalog..."):
                relevant_chunks = chatbot.context_aware_search(
                    question, 
                    conversation_history=st.session_state.messages
                )
                
                answer = chatbot.generate_answer(
                    question, 
                    relevant_chunks, 
                    openai_client,
                    conversation_history=st.session_state.messages
                )
                st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()