import streamlit as st
from openai import OpenAI
import PyPDF2
import fitz  # PyMuPDF - better PDF text extraction
from typing import List, Dict
import json
import os
from datetime import datetime, timedelta
import hashlib
import re
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

class PDFChatbot:
    def __init__(self):
        self.pdf_contents = {}
        self.all_text = ""
        self.text_chunks = []
        self.max_chunk_size = 3000
        self.cache_dir = "pdf_cache"
        self.cache_duration_days = 90  # Cache for 3 months
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Create uploads directory if it doesn't exist
        self.uploads_dir = "uploaded_pdfs"
        if not os.path.exists(self.uploads_dir):
            os.makedirs(self.uploads_dir)
    
    def get_cache_filename(self, identifier: str) -> str:
        """Generate a cache filename based on the PDF files"""
        cache_hash = hashlib.md5(identifier.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"pdf_cache_{cache_hash}.json")
    
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
    
    def save_to_cache(self, identifier: str, pdf_contents: Dict[str, str], text_chunks: List[Dict[str, str]]) -> None:
        """Save processed PDF data to cache"""
        cache_file = self.get_cache_filename(identifier)
        cache_data = {
            'identifier': identifier,
            'cached_at': datetime.now().isoformat(),
            'pdf_contents': pdf_contents,
            'text_chunks': text_chunks
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.warning(f"Could not save cache: {str(e)}")
    
    def load_from_cache(self, identifier: str) -> tuple:
        """Load data from cache if available and valid"""
        cache_file = self.get_cache_filename(identifier)
        
        if not self.is_cache_valid(cache_file):
            return None, None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            pdf_contents = cache_data.get('pdf_contents', {})
            text_chunks = cache_data.get('text_chunks', [])
            
            return pdf_contents, text_chunks
        except Exception as e:
            st.warning(f"Could not load cache: {str(e)}")
            return None, None
    
    def get_cache_info(self, identifier: str) -> Dict:
        """Get information about cached data"""
        cache_file = self.get_cache_filename(identifier)
        
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
                'files_count': len(cache_data.get('pdf_contents', {})),
                'chunks_count': len(cache_data.get('text_chunks', []))
            }
        except:
            return {'exists': False}
    
    def extract_text_from_pdf_pymupdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF (better extraction)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            return text
        except Exception as e:
            st.warning(f"PyMuPDF failed for {pdf_path}: {str(e)}")
            return None
    
    def extract_text_from_pdf_pypdf2(self, pdf_path: str) -> str:
        """Extract text from PDF using PyPDF2 (fallback)"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                return text
        except Exception as e:
            st.warning(f"PyPDF2 failed for {pdf_path}: {str(e)}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using best available method"""
        # Try PyMuPDF first (better extraction)
        text = self.extract_text_from_pdf_pymupdf(pdf_path)
        
        # Fallback to PyPDF2 if PyMuPDF fails
        if not text or len(text.strip()) < 100:
            text = self.extract_text_from_pdf_pypdf2(pdf_path)
        
        if text:
            # Clean up the text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
            text = text.strip()
        
        return text
    
    def process_pdf_files(self, pdf_files: List[str]) -> Dict[str, str]:
        """Process multiple PDF files and extract text"""
        pdf_contents = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_path in enumerate(pdf_files):
            filename = os.path.basename(pdf_path)
            status_text.text(f"Processing: {filename}")
            progress_bar.progress((i + 1) / len(pdf_files))
            
            text = self.extract_text_from_pdf(pdf_path)
            
            if text and len(text.strip()) > 100:  # Only keep files with substantial content
                pdf_contents[filename] = text
                st.success(f"✅ Extracted {len(text)} characters from {filename}")
            else:
                st.warning(f"⚠️ Could not extract meaningful content from {filename}")
        
        status_text.text(f"✅ Processed {len(pdf_contents)} PDF files successfully")
        return pdf_contents
    
    def process_uploaded_files(self, uploaded_files) -> Dict[str, str]:
        """Process uploaded PDF files"""
        pdf_contents = {}
        
        if not uploaded_files:
            return pdf_contents
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing: {uploaded_file.name}")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            try:
                # Save uploaded file temporarily
                temp_path = os.path.join(self.uploads_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract text
                text = self.extract_text_from_pdf(temp_path)
                
                if text and len(text.strip()) > 100:
                    pdf_contents[uploaded_file.name] = text
                    st.success(f"✅ Extracted {len(text)} characters from {uploaded_file.name}")
                else:
                    st.warning(f"⚠️ Could not extract meaningful content from {uploaded_file.name}")
                
                # Clean up temp file
                os.remove(temp_path)
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        status_text.text(f"✅ Processed {len(pdf_contents)} PDF files successfully")
        return pdf_contents
    
    def process_pdf_content(self, pdf_contents: Dict[str, str]) -> None:
        """Process the PDF content into searchable chunks"""
        self.pdf_contents = pdf_contents
        
        # Combine all content
        all_content = []
        for filename, content in pdf_contents.items():
            if content.strip():  # Only add non-empty content
                all_content.append(f"=== From {filename} ===\n{content}\n")
        
        self.all_text = "\n".join(all_content)
        self.text_chunks = self.chunk_text(self.all_text)
        
        st.info(f"Processed {len(pdf_contents)} PDF files into {len(self.text_chunks)} chunks")
    
    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """Split text into manageable chunks with source tracking"""
        chunks = []
        current_filename = None
        
        sections = text.split("=== From ")
        for section in sections:
            if not section.strip():
                continue
                
            if " ===" in section:
                filename_part, content = section.split(" ===", 1)
                current_filename = filename_part.strip()
                text_to_chunk = content.strip()
            else:
                text_to_chunk = section.strip()
            
            if not text_to_chunk:
                continue
            
            # Split into smaller chunks by sentences and paragraphs
            # First split by paragraphs (double newlines)
            paragraphs = text_to_chunk.split('\n\n')
            current_chunk = ""
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # If paragraph is small enough, add it to current chunk
                test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                
                if len(test_chunk) < self.max_chunk_size:
                    current_chunk = test_chunk
                else:
                    # Save current chunk if it has content
                    if current_chunk.strip() and len(current_chunk.strip()) > 100:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'source': current_filename or 'Unknown'
                        })
                    
                    # If this paragraph is too long, split it by sentences
                    if len(paragraph) > self.max_chunk_size:
                        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                        current_chunk = ""
                        
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                            
                            test_sentence_chunk = current_chunk + " " + sentence if current_chunk else sentence
                            
                            if len(test_sentence_chunk) < self.max_chunk_size:
                                current_chunk = test_sentence_chunk
                            else:
                                if current_chunk.strip() and len(current_chunk.strip()) > 100:
                                    chunks.append({
                                        'text': current_chunk.strip(),
                                        'source': current_filename or 'Unknown'
                                    })
                                current_chunk = sentence
                    else:
                        current_chunk = paragraph
            
            # Don't forget the last chunk
            if current_chunk.strip() and len(current_chunk.strip()) > 100:
                chunks.append({
                    'text': current_chunk.strip(),
                    'source': current_filename or 'Unknown'
                })
        
        return chunks
    
    def find_relevant_chunks(self, question: str, chunks: List[Dict[str, str]], max_chunks: int = 5) -> List[Dict[str, str]]:
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
            
            # Look for important question words in the chunk
            for word in question_words:
                if len(word) > 3 and word in chunk_lower:
                    phrase_bonus += 2
                    
            # Look for multi-word phrases
            question_phrases = re.findall(r'\b\w+\s+\w+\b', question_lower)
            for phrase in question_phrases:
                if phrase in chunk_lower:
                    phrase_bonus += 5
            
            total_score = overlap + phrase_bonus
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
            return "I couldn't find relevant information in the PDF files to answer your question. Please try rephrasing or asking about topics covered in the documents."
        
        # Prepare context with sources
        context_parts = []
        sources = []
        for chunk in context_chunks:
            context_parts.append(f"Source: {chunk['source']}\nContent: {chunk['text']}")
            if chunk['source'] not in sources:
                sources.append(chunk['source'])
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant that answers questions based on PDF document content. 

Document Content:
{context}

Question: {question}

Instructions:
- Answer based only on the information provided from the PDF documents
- If the information isn't available, say so clearly
- Be concise but comprehensive
- When referencing specific information, mention which document it came from
- If there are multiple relevant points, organize them clearly
- Use bullet points or numbered lists when appropriate

Answer:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on PDF document content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Add source files at the end
            if sources:
                answer += f"\n\n📚 **Source Documents:**\n" + "\n".join([f"- {source}" for source in sources])
            
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
    st.title("📄 PDF Chatbot")
    st.markdown("Upload PDF files and ask questions about their content!")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
        st.session_state.messages = []
        st.session_state.pdfs_loaded = False
        st.session_state.current_files = []
    
    # Get OpenAI client
    openai_client = get_openai_client()
    
    if not openai_client:
        st.error("❌ OpenAI API key not found!")
        st.info("Please set the OPENAI_API_KEY environment variable or add it to Streamlit secrets.")
        st.stop()
    
    # Sidebar for PDF configuration
    with st.sidebar:
        st.header("📄 PDF Configuration")
        
        # Option 1: Upload PDF files
        st.subheader("📤 Upload PDF Files")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze"
        )
        
        # Option 2: Use local PDF files
        st.subheader("📁 Or Use Local PDF Files")
        pdf_directory = st.text_input(
            "PDF Directory Path",
            placeholder="Enter path to folder containing PDF files",
            help="Path to a folder containing PDF files on your system"
        )
        
        # Check for local PDF files
        local_pdfs = []
        if pdf_directory and os.path.exists(pdf_directory):
            try:
                local_pdfs = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) 
                             if f.lower().endswith('.pdf')]
                if local_pdfs:
                    st.success(f"✅ Found {len(local_pdfs)} PDF files in directory")
                    with st.expander("📋 PDF Files Found"):
                        for pdf in local_pdfs:
                            st.write(f"• {os.path.basename(pdf)}")
                else:
                    st.warning("⚠️ No PDF files found in directory")
            except Exception as e:
                st.error(f"❌ Error reading directory: {str(e)}")
        
        # Generate cache identifier
        cache_identifier = ""
        files_to_process = []
        
        if uploaded_files:
            cache_identifier = "uploaded_" + "_".join([f.name for f in uploaded_files])
            files_to_process = uploaded_files
        elif local_pdfs:
            cache_identifier = "local_" + "_".join([os.path.basename(f) for f in local_pdfs])
            files_to_process = local_pdfs
        
        # Check cache status
        if cache_identifier:
            cache_info = st.session_state.chatbot.get_cache_info(cache_identifier)
            
            if cache_info['exists']:
                if cache_info['is_valid']:
                    st.success(f"✅ Cache available ({cache_info['days_old']} days old)")
                    st.info(f"📊 {cache_info['files_count']} files, {cache_info['chunks_count']} chunks")
                else:
                    st.warning(f"⚠️ Cache expired ({cache_info['days_old']} days old)")
            else:
                st.info("💾 No cache found")
        
        # Process PDFs button
        if st.button("🔄 Process PDFs") and files_to_process:
            # Try to load from cache first
            cached_contents, cached_chunks = st.session_state.chatbot.load_from_cache(cache_identifier)
            
            if cached_contents and cached_chunks:
                st.success("📦 Loaded from cache!")
                st.session_state.chatbot.pdf_contents = cached_contents
                st.session_state.chatbot.text_chunks = cached_chunks
                st.session_state.pdfs_loaded = True
                st.session_state.current_files = list(cached_contents.keys())
                st.info(f"📊 {len(cached_contents)} files, {len(cached_chunks)} chunks loaded from cache")
            else:
                # Process the PDFs
                with st.spinner("Processing PDF files... This may take a few minutes."):
                    try:
                        if uploaded_files:
                            pdf_contents = st.session_state.chatbot.process_uploaded_files(uploaded_files)
                        else:
                            pdf_contents = st.session_state.chatbot.process_pdf_files(local_pdfs)
                        
                        if pdf_contents:
                            # Process the content
                            st.session_state.chatbot.process_pdf_content(pdf_contents)
                            
                            # Save to cache
                            st.session_state.chatbot.save_to_cache(
                                cache_identifier,
                                pdf_contents,
                                st.session_state.chatbot.text_chunks
                            )
                            
                            st.session_state.pdfs_loaded = True
                            st.session_state.current_files = list(pdf_contents.keys())
                            
                            st.success(f"✅ Successfully processed {len(pdf_contents)} PDF files!")
                            st.info(f"📊 Total content: {len(st.session_state.chatbot.text_chunks)} chunks")
                            
                        else:
                            st.error("❌ No content could be extracted from the PDF files")
                    except Exception as e:
                        st.error(f"❌ Error processing PDFs: {str(e)}")
        
        # Force reprocess button
        if st.session_state.pdfs_loaded:
            st.markdown("---")
            if st.button("🔄 Force Reprocess", help="Ignore cache and process fresh"):
                if files_to_process:
                    with st.spinner("Reprocessing PDF files..."):
                        try:
                            if uploaded_files:
                                pdf_contents = st.session_state.chatbot.process_uploaded_files(uploaded_files)
                            else:
                                pdf_contents = st.session_state.chatbot.process_pdf_files(local_pdfs)
                            
                            if pdf_contents:
                                st.session_state.chatbot.process_pdf_content(pdf_contents)
                                st.session_state.chatbot.save_to_cache(
                                    cache_identifier,
                                    pdf_contents,
                                    st.session_state.chatbot.text_chunks
                                )
                                st.success("✅ PDFs reprocessed and cache updated!")
                                st.info(f"📊 {len(pdf_contents)} files, {len(st.session_state.chatbot.text_chunks)} chunks")
                        except Exception as e:
                            st.error(f"❌ Error reprocessing: {str(e)}")
        
        # PDF status
        if st.session_state.pdfs_loaded:
            st.success("📄 PDFs ready for questions!")
            st.caption(f"Files: {len(st.session_state.current_files)}")
            st.caption(f"Chunks: {len(st.session_state.chatbot.text_chunks)}")
            
            # Show loaded files
            with st.expander("📋 Loaded Files"):
                for filename in st.session_state.current_files:
                    st.write(f"• {filename}")
        else:
            st.info("👆 Upload PDFs or specify directory, then click 'Process PDFs'")
    
    # Main chat interface
    if st.session_state.pdfs_loaded:
        st.header("💬 Ask Questions")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if question := st.chat_input("Ask a question about the PDF content..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching PDF content..."):
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
            if st.button("📋 What is the main topic of these documents?"):
                st.session_state.messages.append({"role": "user", "content": "What is the main topic of these documents?"})
                st.rerun()
            
            if st.button("🔍 Summarize the key points"):
                st.session_state.messages.append({"role": "user", "content": "Summarize the key points from all documents"})
                st.rerun()
        
        with col2:
            if st.button("📊 What data or statistics are mentioned?"):
                st.session_state.messages.append({"role": "user", "content": "What data or statistics are mentioned?"})
                st.rerun()
            
            if st.button("❓ What questions are answered in these documents?"):
                st.session_state.messages.append({"role": "user", "content": "What questions are answered in these documents?"})
                st.rerun()
    
    else:
        st.info("👈 Please upload PDF files or specify a directory in the sidebar to get started.")
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()