import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Set
import time
from urllib.parse import urljoin, urlparse
import urllib.robotparser

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

def main():
    st.title("🌐 Website Chatbot")
    st.markdown("Enter a website URL and ask questions about its content!")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = WebsiteChatbot()
        st.session_state.messages = []
        st.session_state.website_loaded = False
        st.session_state.openai_client = None
    
    # Sidebar for API key and website input
    with st.sidebar:
        st.header("⚙️ Setup")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key here"
        )
        
        if api_key:
            st.session_state.openai_client = OpenAI(api_key=api_key)
            st.success("✅ API Key set!")
        else:
            st.warning("⚠️ Please enter your OpenAI API key")
            st.session_state.openai_client = None
        
        st.markdown("---")
        
        # Website input
        st.header("🌐 Website Crawler")
        website_url = st.text_input(
            "Website URL",
            placeholder="https://example.edu",
            help="Enter the college website URL you want to crawl"
        )
        
        max_pages = st.slider(
            "Maximum pages to crawl",
            min_value=1,
            max_value=50,
            value=10,
            help="More pages = more comprehensive but slower"
        )
        
        if website_url and st.button("🕷️ Crawl Website"):
            if not api_key:
                st.error("Please enter your OpenAI API key first!")
            elif not website_url.startswith(('http://', 'https://')):
                st.error("Please enter a valid URL starting with http:// or https://")
            else:
                with st.spinner("Crawling website... This may take a few minutes."):
                    try:
                        # Crawl the website
                        pages_content = st.session_state.chatbot.crawl_website(website_url, max_pages)
                        
                        if pages_content:
                            # Process the content
                            st.session_state.chatbot.process_crawled_content(pages_content)
                            st.session_state.website_loaded = True
                            
                            st.success(f"✅ Successfully crawled {len(pages_content)} pages!")
                            st.info(f"📊 Total content: {len(st.session_state.chatbot.text_chunks)} chunks")
                            
                            # Show crawled pages
                            with st.expander("📄 Crawled Pages"):
                                for url in pages_content.keys():
                                    st.write(f"• {url}")
                        else:
                            st.error("❌ No content could be extracted from the website")
                    except Exception as e:
                        st.error(f"❌ Error crawling website: {str(e)}")
        
        # Website status
        if st.session_state.website_loaded:
            st.success("🌐 Website ready for questions!")
        else:
            st.info("👆 Enter a website URL and crawl it to get started")
    
    # Main chat interface
    if st.session_state.website_loaded and st.session_state.openai_client:
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
                        st.session_state.openai_client
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
    
    elif not st.session_state.website_loaded:
        st.info("👈 Please enter a website URL and crawl it first using the sidebar.")
    elif not st.session_state.openai_client:
        st.warning("👈 Please enter your OpenAI API key in the sidebar.")
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()