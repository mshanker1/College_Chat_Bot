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
import urllib.robotparser
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Ask Scottie - Maryville College Academic Assistant",
    page_icon="🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Configure for pop-out chatbot with Maryville College styling
maryville_popout_style = """
<style>
    /* Import Maryville College colors and fonts */
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
    
    /* Maryville College Official Color Palette */
    :root {
        --maryville-maroon: #5B0F1B;
        --maryville-orange: #EC5E1A;
        --maryville-white: #FFFFFF;
        --maryville-light-gray: #F5F5F5;
        --maryville-dark-gray: #666666;
        --maryville-accent-blue: #1976D2;
        --maryville-success-green: #4CAF50;
        --maryville-warning-gold: #FFC107;
    }
    
    /* Hide the Streamlit header and menu completely */
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0px !important;
        visibility: hidden !important;
    }
    
    /* Hide the footer */
    footer[data-testid="stFooter"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Remove all default Streamlit padding and margins */
    .main .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        background: var(--maryville-white);
        font-family: 'Open Sans', sans-serif;
        min-height: 100vh;
    }
    
    /* Remove any overflow issues */
    .main, .stApp {
        overflow: visible !important;
        background: var(--maryville-white);
    }
    
    /* Hide sidebar completely */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Hide sidebar toggle button */
    button[data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Hide the main menu hamburger */
    #MainMenu {
        display: none !important;
    }
    
    /* Pop-out chatbot container */
    .chatbot-container {
        width: 100%;
        max-width: 400px;
        height: 600px;
        background: var(--maryville-white);
        border-radius: 15px;
        box-shadow: 0 10px 40px rgba(91, 15, 27, 0.2);
        border: 2px solid var(--maryville-maroon);
        display: flex;
        flex-direction: column;
        overflow: hidden;
        position: relative;
    }
    
    /* Chatbot header with logo */
    .chatbot-header {
        background: linear-gradient(135deg, var(--maryville-maroon), var(--maryville-orange));
        color: white;
        padding: 15px 20px;
        text-align: center;
        border-bottom: 2px solid var(--maryville-orange);
    }
    
    .chatbot-header h1 {
        margin: 0;
        font-size: 1.4em;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    
    .chatbot-header p {
        margin: 5px 0 0 0;
        font-size: 0.85em;
        opacity: 0.9;
    }
    
    /* Scottie logo styling */
    .scottie-logo {
        width: 35px;
        height: 35px;
        background: var(--maryville-white);
        border-radius: 50%;
        padding: 5px;
        display: inline-block;
    }
    
    /* Chat area styling */
    .chat-area {
        flex: 1;
        overflow-y: auto;
        padding: 15px;
        background: linear-gradient(180deg, var(--maryville-light-gray) 0%, var(--maryville-white) 100%);
    }
    
    /* Custom scrollbar for chat area */
    .chat-area::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-area::-webkit-scrollbar-track {
        background: var(--maryville-light-gray);
        border-radius: 3px;
    }
    
    .chat-area::-webkit-scrollbar-thumb {
        background: var(--maryville-maroon);
        border-radius: 3px;
    }
    
    .chat-area::-webkit-scrollbar-thumb:hover {
        background: var(--maryville-orange);
    }
    
    /* Chat messages styling */
    .stChatMessage {
        border-radius: 12px;
        border: none;
        margin-bottom: 12px;
        max-width: 85%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* User message styling */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, var(--maryville-maroon), var(--maryville-orange));
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="assistant-message"] {
        background: var(--maryville-white);
        border: 1px solid var(--maryville-light-gray);
        border-left: 4px solid var(--maryville-orange);
        color: var(--maryville-dark-gray);
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    /* Chat input area */
    .stChatFloatingInputContainer {
        background: var(--maryville-white);
        border-top: 2px solid var(--maryville-light-gray);
        padding: 10px 15px;
    }
    
    /* Input styling */
    .stChatInput input {
        border: 2px solid var(--maryville-light-gray);
        border-radius: 25px;
        padding: 10px 15px;
        font-family: 'Open Sans', sans-serif;
        background: var(--maryville-white);
    }
    
    .stChatInput input:focus {
        border-color: var(--maryville-maroon);
        box-shadow: 0 0 0 2px rgba(91, 15, 27, 0.2);
        outline: none;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--maryville-maroon), var(--maryville-orange));
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Open Sans', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        margin-bottom: 5px;
        padding: 8px 12px;
        font-size: 0.85em;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--maryville-orange), var(--maryville-maroon));
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(91, 15, 27, 0.3);
    }
    
    /* Compact info sections */
    .info-section {
        background: var(--maryville-white);
        border: 1px solid var(--maryville-light-gray);
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        font-size: 0.85em;
    }
    
    .info-section h3 {
        color: var(--maryville-maroon);
        margin: 0 0 8px 0;
        font-size: 1em;
    }
    
    /* Source indicators */
    .source-pdf {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border-left: 3px solid var(--maryville-accent-blue);
    }
    
    .source-web {
        background: linear-gradient(135deg, #E8F5E8, #C8E6C9);
        border-left: 3px solid var(--maryville-success-green);
    }
    
    /* Responsive adjustments */
    @media (max-width: 480px) {
        .chatbot-container {
            max-width: 100%;
            height: 100vh;
            border-radius: 0;
            border: none;
        }
        
        .chatbot-header h1 {
            font-size: 1.2em;
        }
        
        .scottie-logo {
            width: 30px;
            height: 30px;
        }
    }
    
    /* Hide Streamlit branding */
    .stDeployButton {
        display: none !important;
    }
    
    footer {
        display: none !important;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background: var(--maryville-success-green);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
"""

# Apply pop-out chatbot styling conditionally
# Check if we're in embedded mode
try:
    query_params = st.query_params
    is_embedded = query_params.get("embed") or query_params.get("embedded")
except AttributeError:
    # Fallback for older Streamlit versions
    try:
        query_params = st.experimental_get_query_params()
        is_embedded = query_params.get("embed") or query_params.get("embedded")
    except:
        is_embedded = False

# Apply different styles based on mode
if is_embedded:
    st.markdown(maryville_popout_style, unsafe_allow_html=True)
else:
    # Standalone mode - more visible styling
    standalone_style = """
    <style>
        /* Import Maryville College colors and fonts */
        @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
        
        /* Maryville College Official Color Palette */
        :root {
            --maryville-maroon: #5B0F1B;
            --maryville-orange: #EC5E1A;
            --maryville-white: #FFFFFF;
            --maryville-light-gray: #F5F5F5;
            --maryville-dark-gray: #666666;
            --maryville-accent-blue: #1976D2;
            --maryville-success-green: #4CAF50;
            --maryville-warning-gold: #FFC107;
        }
        
        /* Main container styling for standalone */
        .main .block-container {
            max-width: 800px;
            padding: 2rem 1rem;
            background: linear-gradient(135deg, var(--maryville-light-gray) 0%, var(--maryville-white) 100%);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(91, 15, 27, 0.1);
            border: 2px solid var(--maryville-maroon);
            font-family: 'Open Sans', sans-serif;
        }
        
        /* Header styling */
        .main h1 {
            color: var(--maryville-maroon);
            text-align: center;
            font-family: 'Open Sans', sans-serif;
            font-weight: 700;
            border-bottom: 3px solid var(--maryville-orange);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        /* Chat messages styling */
        .stChatMessage {
            border-radius: 12px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* User message styling */
        .stChatMessage[data-testid="user-message"] {
            background: linear-gradient(135deg, var(--maryville-maroon), var(--maryville-orange));
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        /* Assistant message styling */
        .stChatMessage[data-testid="assistant-message"] {
            background: var(--maryville-white);
            border: 1px solid var(--maryville-light-gray);
            border-left: 4px solid var(--maryville-orange);
            color: var(--maryville-dark-gray);
            border-bottom-left-radius: 4px;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--maryville-maroon), var(--maryville-orange));
            color: white;
            border: none;
            border-radius: 8px;
            font-family: 'Open Sans', sans-serif;
            font-weight: 600;
            transition: all 0.3s ease;
            margin-bottom: 5px;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, var(--maryville-orange), var(--maryville-maroon));
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(91, 15, 27, 0.3);
        }
        
        /* Info sections */
        .info-section {
            background: var(--maryville-white);
            border: 1px solid var(--maryville-light-gray);
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .info-section h3 {
            color: var(--maryville-maroon);
            margin: 0 0 10px 0;
        }
        
        /* Source indicators */
        .source-pdf {
            background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
            border-left: 3px solid var(--maryville-accent-blue);
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 4px;
        }
        
        .source-web {
            background: linear-gradient(135deg, #E8F5E8, #C8E6C9);
            border-left: 3px solid var(--maryville-success-green);
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 4px;
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
            background: var(--maryville-success-green);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
    """
    st.markdown(standalone_style, unsafe_allow_html=True)

# --- INSTITUTIONAL CONFIGURATION ---
class Config:
    """Optimized configuration for institutional deployment."""
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

# --- HELPER FUNCTIONS & CLASSES ---

class InstitutionalPDFChatbot:
    """Optimized PDF chatbot for institutional deployment with pre-loaded documents."""
    
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
        
        # Load the sentence transformer model with optimized settings
        self.embedding_model = self._load_embedding_model()
        if self.embedding_model:
            self.model_device = self.embedding_model.device
        else:
            st.error("Could not load embedding model.")
            st.stop()

    @st.cache_resource
    def _load_embedding_model(_self):
        """Load embedding model with optimized retry logic."""
        for attempt in range(Config.MAX_RETRIES):
            try:
                # Determine best device
                if torch.cuda.is_available():
                    device = 'cuda'
                elif torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
                
                # Model cache directory
                cache_folder = os.path.join(os.getcwd(), "model_cache")
                os.makedirs(cache_folder, exist_ok=True)
                
                # Load model
                model = SentenceTransformer(
                    Config.MODEL_NAME, 
                    device=device,
                    cache_folder=cache_folder
                )
                
                return model
                
            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    wait_time = (2 ** attempt) * 3
                    if attempt < Config.MAX_RETRIES - 1:
                        time.sleep(wait_time)
                else:
                    break
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    try:
                        return SentenceTransformer('paraphrase-MiniLM-L6-v2')
                    except:
                        return None
                else:
                    time.sleep(2 ** attempt)
        
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
        """Enhanced PDF text extraction with better cleaning."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Better text cleaning
                page_text = re.sub(r'\s+', ' ', page_text)
                page_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', page_text)
                page_text = re.sub(r'^\d+\s*$', '', page_text, flags=re.MULTILINE)
                
                text += page_text + "\n\n"
            
            doc.close()
            
            # Final cleaning
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
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text from main content areas
            text_content = ""
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
            text_content = re.sub(r'\n+', '\n', text_content)
            
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
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error fetching {url}: {e}")
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
        
        # Academic requirement patterns that need larger chunks
        requirement_patterns = [
            r'General Education Requirements',
            r'Category [IVX]+ Domains?',
            r'Foundations?\s*\(\d+\s*Credit Hours?\)',
            r'Required courses? include:?',
            r'Completion of one',
            r'The Maryville Curriculum',
            r'Undergraduate Degree Requirements',
            r'Domain[s]? of Knowledge',
            r'Empirical Study',
            r'Scientific Reasoning',
            r'Mathematical Reasoning',
            r'Creative Arts',
            r'Ethical Citizenship'
        ]
        
        # Check if this text contains academic requirements
        is_requirements_section = any(re.search(pattern, text, re.IGNORECASE) 
                                     for pattern in requirement_patterns)
        
        # Adjust chunking parameters based on content type
        if is_requirements_section:
            chunk_size = 1400
            overlap_size = 300
        else:
            chunk_size = Config.CHUNK_SIZE
            overlap_size = Config.CHUNK_OVERLAP
        
        # Split into sentences first
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
        
        # Add the last chunk
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

    def extract_context_keywords(self, conversation_history: List[Dict]) -> List[str]:
        """Extract important keywords from recent conversation for enhanced search."""
        if not conversation_history:
            return []
        
        recent_messages = conversation_history[-6:]
        
        keywords = []
        for msg in recent_messages:
            if msg['role'] == 'assistant':
                quotes = re.findall(r'"([^"]*)"', msg['content'])
                keywords.extend(quotes)
                
                doc_refs = re.findall(r'\[([^\]]+\.pdf)\]', msg['content'])
                keywords.extend(doc_refs)
                
                bold_terms = re.findall(r'\*\*([^*]+)\*\*', msg['content'])
                keywords.extend(bold_terms)
        