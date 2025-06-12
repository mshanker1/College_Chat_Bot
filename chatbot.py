import streamlit as st
import PyPDF2
from openai import OpenAI
import os
from io import BytesIO
import re
from typing import List, Dict
import time

# Configure page
st.set_page_config(
    page_title="College Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

class PDFChatbot:
    def __init__(self):
        self.document_text = ""
        self.document_chunks = []
        self.max_chunk_size = 3000
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers/footers (basic cleanup)
        text = re.sub(r'Page \d+', '', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks for processing"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < self.max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def find_relevant_chunks(self, question: str, chunks: List[str], max_chunks: int = 3) -> List[str]:
        """Find the most relevant chunks for the question (simple keyword matching)"""
        question_words = set(question.lower().split())
        chunk_scores = []
        
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            # Simple relevance score based on word overlap
            overlap = len(question_words.intersection(chunk_words))
            chunk_scores.append((i, overlap, chunk))
        
        # Sort by relevance and return top chunks
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for _, _, chunk in chunk_scores[:max_chunks] if chunk_scores[0][1] > 0]
    
    def generate_answer(self, question: str, context_chunks: List[str], client: OpenAI) -> str:
        """Generate answer using OpenAI API"""
        if not context_chunks:
            return "I couldn't find relevant information in the document to answer your question. Please try rephrasing or asking about topics covered in the uploaded document."
        
        context = "\n\n".join(context_chunks)
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided document content. 

Document Content:
{context}

Question: {question}

Instructions:
- Answer based only on the information provided in the document content
- If the information isn't in the document, say so clearly
- Be concise but comprehensive
- If you reference specific information, mention it's from the document
- If there are multiple relevant points, organize them clearly

Answer:"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on document content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}. Please check your OpenAI API key and try again."

def main():
    st.title("🤖 College Document Chatbot")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
        st.session_state.messages = []
        st.session_state.document_loaded = False
        st.session_state.openai_client = None
    
    # Sidebar for API key and file upload
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
        
        # File upload
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload the college document you want to ask questions about"
        )
        
        if uploaded_file is not None:
            if st.button("📊 Process Document"):
                if not api_key:
                    st.error("Please enter your OpenAI API key first!")
                else:
                    with st.spinner("Processing PDF..."):
                        # Extract and process text
                        raw_text = st.session_state.chatbot.extract_text_from_pdf(uploaded_file)
                        if raw_text:
                            cleaned_text = st.session_state.chatbot.clean_text(raw_text)
                            chunks = st.session_state.chatbot.chunk_text(cleaned_text)
                            
                            st.session_state.chatbot.document_text = cleaned_text
                            st.session_state.chatbot.document_chunks = chunks
                            st.session_state.document_loaded = True
                            
                            st.success(f"✅ Document processed successfully!")
                            st.info(f"📊 Document stats:\n- {len(chunks)} chunks created\n- {len(cleaned_text)} characters")
                        else:
                            st.error("Failed to extract text from PDF")
        
        # Document status
        if st.session_state.document_loaded:
            st.success("📄 Document ready for questions!")
        else:
            st.info("👆 Upload and process a PDF to get started")
    
    # Main chat interface
    if st.session_state.document_loaded and st.session_state.openai_client:
        st.header("💬 Ask Questions")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if question := st.chat_input("Ask a question about the document..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Find relevant chunks
                    relevant_chunks = st.session_state.chatbot.find_relevant_chunks(
                        question, 
                        st.session_state.chatbot.document_chunks
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
            if st.button("📚 What are the admission requirements?"):
                st.session_state.messages.append({"role": "user", "content": "What are the admission requirements?"})
                st.rerun()
        
        with col2:
            if st.button("📅 When does the semester start?"):
                st.session_state.messages.append({"role": "user", "content": "When does the semester start?"})
                st.rerun()
    
    elif not st.session_state.document_loaded:
        st.info("👈 Please upload and process a PDF document first using the sidebar.")
    elif not st.session_state.openai_client:
        st.warning("👈 Please enter your OpenAI API key in the sidebar.")
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()