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
    
    def get_cache_filename(self, url: str) -> str:
        """Generate a cache filename based on the URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"cache_{url_hash}.json")
    
    def is_cache_valid(self, cache_file: str) -> bool:
        """Check if cache file exists and is still valid"""
        if not os.path.exists(cache_file):
            return False
        
        try:
            with open(cache_file, 'r') as f:
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
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            st.warning(f"Could not save cache: {str(e)}")
    
    def load_from_cache(self, url: str) -> tuple:
        """Load data from cache if available and valid"""
        cache_file = self.get_cache_filename(url)
        
        if not self.is_cache_valid(cache_file):
            return None, None
        
        try:
            with open(cache_file, 'r') as f:
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
            with open(cache_file, 'r') as f:
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
    
    def get_page_content(self, url: str) -> str:
        """Extract text content from a single webpage"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            st.warning(f"Could not crawl {url}: {str(e)}")
            return ""
    
    def find_internal_links(self, base_url: str, html_content: str) -> Set[str]:
        """Find internal links from the page"""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Only include internal links
            if urlparse(full_url).netloc == base_domain:
                # Clean URL (remove fragments)
                clean_url = full_url.split('#')[0]
                if clean_url not in links and clean_url != base_url:
                    links.add(clean_url)
        
        return links
    
    def crawl_website(self, start_url: str, max_pages: int = None) -> Dict[str, str]:
        """Crawl website starting from the given URL"""
        if max_pages is None:
            max_pages = self.max_pages
            
        # Check robots.txt
        if not self.check_robots_txt(start_url):
            st.warning("⚠️ Robots.txt disallows crawling this site. Proceeding with single page only.")
            max_pages = 1
        
        visited = set()
        to_visit = {start_url}
        pages_content = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while to_visit and len(visited) < max_pages:
            current_url = to_visit.pop()
            if current_url in visited:
                continue
                
            status_text.text(f"Crawling: {current_url}")
            progress_bar.progress(len(visited) / max_pages)
            
            # Get page content
            page_text = self.get_page_content(current_url)
            if page_text:
                pages_content[current_url] = page_text
                visited.add(current_url)
                
                # Find more links only if we haven't reached the limit
                if len(visited) < max_pages:
                    try:
                        response = requests.get(current_url, timeout=10)
                        new_links = self.find_internal_links(current_url, response.text)
                        # Add new links but don't exceed our limit
                        for link in new_links:
                            if link not in visited and len(to_visit) + len(visited) < max_pages:
                                to_visit.add(link)
                    except:
                        pass
                
                # Be respectful - add a small delay
                time.sleep(0.5)
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Crawled {len(pages_content)} pages")
        
        return pages_content
    
    def process_crawled_content(self, pages_content: Dict[str, str]) -> None:
        """Process the crawled content into searchable chunks"""
        self.pages_content = pages_content
        
        # Combine all content
        all_content = []
        for url, content in pages_content.items():
            all_content.append(f"=== From {url} ===\n{content}\n")
        
        self.all_text = "\n".join(all_content)
        self.text_chunks = self.chunk_text(self.all_text)
    
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
            
            # Split into smaller chunks
            sentences = text_to_chunk.split('. ')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) < self.max_chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        chunks.append({
                            'text': current_chunk.strip(),
                            'source': current_url or 'Unknown'
                        })
                    current_chunk = sentence + ". "
            
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'source': current_url or 'Unknown'
                })
        
        return chunks
    
    def find_relevant_chunks(self, question: str, chunks: List[Dict[str, str]], max_chunks: int = 3) -> List[Dict[str, str]]:
        """Find the most relevant chunks for the question"""
        question_words = set(question.lower().split())
        chunk_scores = []
        
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk['text'].lower().split())
            # Simple relevance score based on word overlap
            overlap = len(question_words.intersection(chunk_words))
            chunk_scores.append((i, overlap, chunk))
        
        # Sort by relevance and return top chunks
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for _, _, chunk in chunk_scores[:max_chunks] if chunk_scores[0][1] > 0]
    
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
            max_value=5000,
            value=10,
            help="More pages = more comprehensive but slower"
        )
        
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
                                    for url in pages_content.keys():
                                        st.write(f"• {url}")
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
                    except Exception as e:
                        st.error(f"❌ Error recrawling: {str(e)}")
        
        # Website status
        if st.session_state.website_loaded:
            st.success("🌐 Website ready for questions!")
            st.caption(f"Current: {st.session_state.current_url}")
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
