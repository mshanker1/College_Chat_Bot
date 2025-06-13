import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Set
import time
from urllib.parse import urljoin, urlparse
import urllib.robotparser
import json
import os
from datetime import datetime, timedelta
import hashlib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure page
st.set_page_config(
    page_title="Website Chatbot",
    page_icon="🌐",
    layout="wide"
)

class WebsiteChatbot:
    def __init__(self):
        self.pages_content = {}
        self.all_text = ""
        self.text_chunks = []
        self.max_chunk_size = 3000
        self.max_pages = 20
        self.cache_dir = "website_cache"
        self.cache_duration_days = 90  # Cache for 3 months
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def get_cache_filename(self, url: str) -> str:
        """Generate a cache filename based on the URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"cache_{url_hash}.json")
    
    def is_cache_valid(self, cache_file: str) -> bool:
        """Check if cache file exists and is still valid"""
        if not os.path.exists(cache_file):
            return False
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            cached_date = datetime.fromisoformat(cache_data.get('cached_at', ''))
            return datetime.now() - cached_date < timedelta(days=self.cache_duration_days)
        except:
            return False
    
    def save_to_cache(self, url: str, pages_content: Dict[str, str], text_chunks: List[Dict[str, str]]) -> None:
        """Save crawled data to cache"""
        cache_file = self.get_cache_filename(url)
        cache_data = {
            'url': url,
            'cached_at': datetime.now().isoformat(),
            'pages_content': pages_content,
            'text_chunks': text_chunks
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.warning(f"Could not save cache: {str(e)}")
    
    def load_from_cache(self, url: str) -> tuple:
        """Load data from cache if available and valid"""
        cache_file = self.get_cache_filename(url)
        
        if not self.is_cache_valid(cache_file):
            return None, None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            pages_content = cache_data.get('pages_content', {})
            text_chunks = cache_data.get('text_chunks', [])
            
            return pages_content, text_chunks
        except Exception as e:
            st.warning(f"Could not load cache: {str(e)}")
            return None, None
    
    def get_cache_info(self, url: str) -> Dict:
        """Get information about cached data"""
        cache_file = self.get_cache_filename(url)
        
        if not os.path.exists(cache_file):
            return {'exists': False}
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            cached_date = datetime.fromisoformat(cache_data.get('cached_at', ''))
            days_old = (datetime.now() - cached_date).days
            is_valid = days_old < self.cache_duration_days
            
            return {
                'exists': True,
                'cached_at': cached_date,
                'days_old': days_old,
                'is_valid': is_valid,
                'pages_count': len(cache_data.get('pages_content', {})),
                'chunks_count': len(cache_data.get('text_chunks', []))
            }
        except:
            return {'exists': False}
        
    def check_robots_txt(self, base_url: str) -> bool:
        """Check if crawling is allowed by robots.txt"""
        try:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(urljoin(base_url, '/robots.txt'))
            rp.read()
            return rp.can_fetch('*', base_url)
        except:
            return True  # If robots.txt can't be read, assume crawling is allowed
    
    def is_valid_content_type(self, url: str) -> bool:
        """Check if URL points to HTML content"""
        try:
            response = self.session.head(url, timeout=5)
            content_type = response.headers.get('content-type', '').lower()
            return 'text/html' in content_type or content_type == ''
        except:
            return True  # Assume it's valid if we can't check
    
    def should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped"""
        url_lower = url.lower()
        skip_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                          '.jpg', '.jpeg', '.png', '.gif', '.svg', '.mp4', '.avi', 
                          '.zip', '.rar', '.exe', '.dmg'}
        
        skip_patterns = ['mailto:', 'tel:', 'javascript:', 'ftp:', '#']
        
        # Check for skip patterns
        for pattern in skip_patterns:
            if pattern in url_lower:
                return True
        
        # Check for file extensions
        for ext in skip_extensions:
            if url_lower.endswith(ext):
                return True
        
        return False
    
    def get_page_content(self, url: str) -> str:
        """Extract text content from a single webpage"""
        try:
            if self.should_skip_url(url):
                return ""
            
            if not self.is_valid_content_type(url):
                return ""
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # Check if content is actually HTML
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type and content_type != '':
                return ""
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", 
                               "aside", "noscript", "iframe", "form"]):
                element.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text more thoroughly
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk and len(chunk) > 2)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            return text
        except requests.exceptions.RequestException as e:
            st.warning(f"Network error crawling {url}: {str(e)}")
            return ""
        except Exception as e:
            st.warning(f"Error crawling {url}: {str(e)}")
            return ""
    
    def find_internal_links(self, base_url: str, html_content: str) -> Set[str]:
        """Find internal links from the page"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = set()
            base_domain = urlparse(base_url).netloc
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Skip obvious non-page links
                if any(skip in href for skip in ['mailto:', 'tel:', 'javascript:', '#', 'ftp:']):
                    continue
                
                full_url = urljoin(base_url, href)
                parsed_url = urlparse(full_url)
                
                # Only include internal links
                if parsed_url.netloc == base_domain:
                    # Clean URL (remove fragments and query parameters for deduplication)
                    clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                    
                    # Remove trailing slash for consistency
                    if clean_url.endswith('/') and len(clean_url) > 1:
                        clean_url = clean_url[:-1]
                    
                    if clean_url and clean_url != base_url.rstrip('/'):
                        links.add(clean_url)
            
            return links
        except Exception as e:
            st.warning(f"Error finding links in {base_url}: {str(e)}")
            return set()
    
    def crawl_website(self, start_url: str, max_pages: int = None) -> Dict[str, str]:
        """Crawl website starting from the given URL"""
        if max_pages is None:
            max_pages = self.max_pages
        
        st.info(f"Starting crawl with max_pages = {max_pages}")
        
        # Clean start URL
        start_url = start_url.rstrip('/')
        
        # Check robots.txt
        if not self.check_robots_txt(start_url):
            st.warning("⚠️ Robots.txt disallows crawling this site. Proceeding with single page only.")
            max_pages = 1
        
        visited = set()
        to_visit = {start_url}
        pages_content = {}
        failed_urls = set()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            while to_visit and len(visited) < max_pages:
                current_url = to_visit.pop()
                
                if current_url in visited or current_url in failed_urls:
                    continue
                    
                status_text.text(f"Crawling ({len(visited)+1}/{max_pages}): {current_url}")
                
                # Get page content
                page_text = self.get_page_content(current_url)
                if page_text and len(page_text.strip()) > 100:  # Only keep pages with substantial content
                    pages_content[current_url] = page_text
                    visited.add(current_url)
                    
                    # Update progress
                    progress_bar.progress(len(visited) / max_pages)
                    
                    # Find more links only if we haven't reached the limit
                    if len(visited) < max_pages:
                        try:
                            response = self.session.get(current_url, timeout=10)
                            new_links = self.find_internal_links(current_url, response.text)
                            
                            # Add new links but prioritize by depth/relevance
                            links_to_add = []
                            for link in new_links:
                                if (link not in visited and 
                                    link not in failed_urls and 
                                    link not in to_visit and
                                    not self.should_skip_url(link)):
                                    links_to_add.append(link)
                            
                            # Limit the number of new links to prevent explosion
                            max_new_links = min(20, max_pages - len(visited) - len(to_visit))
                            for link in links_to_add[:max_new_links]:
                                to_visit.add(link)
                                
                        except Exception as e:
                            st.warning(f"Error getting links from {current_url}: {str(e)}")
                else:
                    failed_urls.add(current_url)
                
                # Be respectful - add a delay
                time.sleep(0.5)
                
                # Show progress update
                if len(visited) % 5 == 0:
                    st.info(f"Progress: {len(visited)} pages crawled, {len(to_visit)} pages in queue")
        
        except KeyboardInterrupt:
            st.warning("Crawling interrupted by user")
        except Exception as e:
            st.error(f"Crawling error: {str(e)}")
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Crawled {len(pages_content)} pages successfully")
        
        if failed_urls:
            st.warning(f"⚠️ Failed to crawl {len(failed_urls)} URLs")
        
        return pages_content
    
    def process_crawled_content(self, pages_content: Dict[str, str]) -> None:
        """Process the crawled content into searchable chunks"""
        self.pages_content = pages_content
        
        # Combine all content
        all_content = []
        for url, content in pages_content.items():
            if content.strip():  # Only add non-empty content
                all_content.append(f"=== From {url} ===\n{content}\n")
        
        self.all_text = "\n".join(all_content)
        self.text_chunks = self.chunk_text(self.all_text)
        
        st.info(f"Processed {len(pages_content)} pages into {len(self.text_chunks)} chunks")
    
    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """Split text into manageable chunks with source tracking"""
        chunks = []
        current_url = None
        
        sections = text.split("=== From ")
        for section in sections:
            if not section.strip():
                continue
                
            if " ===" in section:
                url_part, content = section.split(" ===", 1)
                current_url = url_part.strip()
                text_to_chunk = content.strip()
            else:
                text_to_chunk = section.strip()
            
            if not text_to_chunk:
                continue
            
            # Split into smaller chunks by sentences
            sentences = re.split(r'(?<=[.!?])\s+', text_to_chunk)
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Check if adding this sentence would exceed chunk size
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(test_chunk) < self.max_chunk_size:
                    current_chunk = test_chunk
                else:
                    # Save current chunk if it has content
                    if current_chunk.strip() and len(current_chunk.strip()) > 50:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'source': current_url or 'Unknown'
                        })
                    # Start new chunk with current sentence
                    current_chunk = sentence
            
            # Don't forget the last chunk
            if current_chunk.strip() and len(current_chunk.strip()) > 50:
                chunks.append({
                    'text': current_chunk.strip(),
                    'source': current_url or 'Unknown'
                })
        
        return chunks
    
    def find_relevant_chunks(self, question: str, chunks: List[Dict[str, str]], max_chunks: int = 3) -> List[Dict[str, str]]:
        """Find the most relevant chunks for the question"""
        if not chunks:
            return []
            
        question_words = set(re.findall(r'\w+', question.lower()))
        chunk_scores = []
        
        for i, chunk in enumerate(chunks):
            chunk_words = set(re.findall(r'\w+', chunk['text'].lower()))
            
            # Calculate relevance score
            overlap = len(question_words.intersection(chunk_words))
            
            # Bonus for exact phrase matches
            question_lower = question.lower()
            chunk_lower = chunk['text'].lower()
            phrase_bonus = 0
            for word in question_words:
                if len(word) > 3 and word in chunk_lower:
                    phrase_bonus += 1
            
            total_score = overlap + phrase_bonus * 2
            chunk_scores.append((i, total_score, chunk))
        
        # Sort by relevance and return top chunks
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Only return chunks with meaningful overlap
        relevant_chunks = []
        for _, score, chunk in chunk_scores[:max_chunks]:
            if score > 0:
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, str]], client: OpenAI) -> str:
        """Generate answer using OpenAI API with source attribution"""
        if not context_chunks:
            return "I couldn't find relevant information on the website to answer your question. Please try rephrasing or asking about topics covered on the website."
        
        # Prepare context with sources
        context_parts = []
        sources = []
        for chunk in context_chunks:
            context_parts.append(f"Source: {chunk['source']}\nContent: {chunk['text']}")
            if chunk['source'] not in sources:
                sources.append(chunk['source'])
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant that answers questions based on website content. 

Website Content:
{context}

Question: {question}

Instructions:
- Answer based only on the information provided from the website
- If the information isn't available, say so clearly
- Be concise but comprehensive
- When referencing specific information, mention which page it came from
- If there are multiple relevant points, organize them clearly

Answer:"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on website content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Add source links at the end
            if sources:
                answer += f"\n\n📚 **Sources:**\n" + "\n".join([f"- {source}" for source in sources])
            
            return answer
        except Exception as e:
            return f"Error generating response: {str(e)}. Please check your OpenAI API key and try again."

def get_openai_client():
    """Get OpenAI client with API key from environment or secrets"""
    api_key = None
    
    # Try to get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If not found, try Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            pass
    
    if api_key:
        return OpenAI(api_key=api_key)
    else:
        return None

def main():
    st.title("🌐 Website Chatbot")
    st.markdown("Ask questions about pre-configured website content!")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = WebsiteChatbot()
        st.session_state.messages = []
        st.session_state.website_loaded = False
        st.session_state.current_url = None
    
    # Get OpenAI client
    openai_client = get_openai_client()
    
    if not openai_client:
        st.error("❌ OpenAI API key not found!")
        st.info("Please set the OPENAI_API_KEY environment variable or add it to Streamlit secrets.")
        st.stop()
    
    # Sidebar for website configuration
    with st.sidebar:
        st.header("🌐 Website Configuration")
        
        # Pre-configured website URL (you can change this)
        default_url = "https://www.maryvillecollege.edu/"  # Replace with your target website
        website_url = st.text_input(
            "Website URL",
            value=default_url,
            help="Website to crawl and cache"
        )
        
        max_pages = st.slider(
            "Maximum pages to crawl",
            min_value=1,
            max_value=500,
            value=50,
            help="More pages = more comprehensive but slower"
        )
        
        # Update the chatbot's max_pages setting
        if 'chatbot' in st.session_state:
            st.session_state.chatbot.max_pages = max_pages
        
        # Check cache status
        if website_url:
            cache_info = st.session_state.chatbot.get_cache_info(website_url)
            
            if cache_info['exists']:
                if cache_info['is_valid']:
                    st.success(f"✅ Cache available ({cache_info['days_old']} days old)")
                    st.info(f"📊 {cache_info['pages_count']} pages, {cache_info['chunks_count']} chunks")
                else:
                    st.warning(f"⚠️ Cache expired ({cache_info['days_old']} days old)")
            else:
                st.info("💾 No cache found")
        
        # Load website button
        if st.button("🔄 Load Website"):
            if not website_url.startswith(('http://', 'https://')):
                st.error("Please enter a valid URL starting with http:// or https://")
            else:
                # Try to load from cache first
                cached_pages, cached_chunks = st.session_state.chatbot.load_from_cache(website_url)
                
                if cached_pages and cached_chunks:
                    st.success("📦 Loaded from cache!")
                    st.session_state.chatbot.pages_content = cached_pages
                    st.session_state.chatbot.text_chunks = cached_chunks
                    st.session_state.website_loaded = True
                    st.session_state.current_url = website_url
                    st.info(f"📊 {len(cached_pages)} pages, {len(cached_chunks)} chunks loaded from cache")
                else:
                    # Cache not available or expired, crawl the website
                    with st.spinner("Crawling website... This may take a few minutes."):
                        try:
                            # Crawl the website
                            pages_content = st.session_state.chatbot.crawl_website(website_url, max_pages)
                            
                            if pages_content:
                                # Process the content
                                st.session_state.chatbot.process_crawled_content(pages_content)
                                
                                # Save to cache
                                st.session_state.chatbot.save_to_cache(
                                    website_url, 
                                    pages_content, 
                                    st.session_state.chatbot.text_chunks
                                )
                                
                                st.session_state.website_loaded = True
                                st.session_state.current_url = website_url
                                
                                st.success(f"✅ Successfully crawled and cached {len(pages_content)} pages!")
                                st.info(f"📊 Total content: {len(st.session_state.chatbot.text_chunks)} chunks")
                                
                                # Show crawled pages
                                with st.expander("📄 Crawled Pages"):   
                                    for i, url in enumerate(pages_content.keys(), 1):
                                        st.write(f"{i}. {url}")
                            else:
                                st.error("❌ No content could be extracted from the website")
                        except Exception as e:
                            st.error(f"❌ Error crawling website: {str(e)}")
        
        # Force recrawl button
        if st.session_state.website_loaded:
            st.markdown("---")
            if st.button("🔄 Force Recrawl", help="Ignore cache and crawl fresh"):
                with st.spinner("Recrawling website..."):
                    try:
                        pages_content = st.session_state.chatbot.crawl_website(website_url, max_pages)
                        if pages_content:
                            st.session_state.chatbot.process_crawled_content(pages_content)
                            st.session_state.chatbot.save_to_cache(
                                website_url, 
                                pages_content, 
                                st.session_state.chatbot.text_chunks
                            )
                            st.success("✅ Website recrawled and cache updated!")
                            st.info(f"📊 {len(pages_content)} pages, {len(st.session_state.chatbot.text_chunks)} chunks")
                    except Exception as e:
                        st.error(f"❌ Error recrawling: {str(e)}")
        
        # Website status
        if st.session_state.website_loaded:
            st.success("🌐 Website ready for questions!")
            st.caption(f"Current: {st.session_state.current_url}")
            st.caption(f"Pages: {len(st.session_state.chatbot.pages_content)}")
            st.caption(f"Chunks: {len(st.session_state.chatbot.text_chunks)}")
        else:
            st.info("👆 Click 'Load Website' to get started")
    
    # Main chat interface
    if st.session_state.website_loaded:
        st.header("💬 Ask Questions")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if question := st.chat_input("Ask a question about the website..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching website content..."):
                    # Find relevant chunks
                    relevant_chunks = st.session_state.chatbot.find_relevant_chunks(
                        question, 
                        st.session_state.chatbot.text_chunks
                    )
                    
                    # Generate answer
                    answer = st.session_state.chatbot.generate_answer(
                        question, 
                        relevant_chunks,
                        openai_client
                    )
                    
                    st.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Example questions
        st.markdown("---")
        st.subheader("💡 Example Questions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎓 What programs does the college offer?"):
                st.session_state.messages.append({"role": "user", "content": "What programs does the college offer?"})
                st.rerun()
            
            if st.button("💰 What are the tuition costs?"):
                st.session_state.messages.append({"role": "user", "content": "What are the tuition costs?"})
                st.rerun()
        
        with col2:
            if st.button("📅 What are the admission deadlines?"):
                st.session_state.messages.append({"role": "user", "content": "What are the admission deadlines?"})
                st.rerun()
            
            if st.button("🏠 What housing options are available?"):
                st.session_state.messages.append({"role": "user", "content": "What housing options are available?"})
                st.rerun()
    
    else:
        st.info("👈 Please click 'Load Website' in the sidebar to get started.")
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()