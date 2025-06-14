import streamlit as st
# from openai import OpenAI # Removed OpenAI import
from huggingface_hub import InferenceClient # New import for Hugging Face
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import os
import json
from datetime import datetime, timedelta
import hashlib
from typing import List, Dict, Any # Added Any for generic client type hint
import torch # Import torch explicitly for device handling

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Advanced PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- HELPER FUNCTIONS & CLASSES ---

class PDFChatbot:
    """A class to handle PDF processing, chunking, embedding, and answering questions."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.pdf_contents: Dict[str, str] = {}
        self.text_chunks: List[Dict] = []
        self.chunk_embeddings = None
        self.uploads_dir = "uploaded_pdfs"
        self.cache_dir = "pdf_cache"
        self.cache_duration_days = 90
        
        # Initialize directories
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load the sentence transformer model
        # st.cache_resource is used here to avoid re-loading the model on every rerun
        # This significantly improves performance.
        @st.cache_resource
        def load_embedding_model(name):
            try:
                # Specify device 'cuda' if GPU is available, 'mps' for Apple Silicon, else 'cpu'
                # SentenceTransformer automatically handles this if not specified, but for clarity
                # and explicit control, we can set it.
                if torch.cuda.is_available():
                    device = 'cuda'
                elif torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
                return SentenceTransformer(name, device=device)
            except Exception as e:
                st.error(f"Error loading embedding model '{name}': {e}. Please ensure you have internet access or the model is cached locally.")
                st.stop()
        self.embedding_model = load_embedding_model(model_name)
        # Store the model's actual device for later use
        self.model_device = self.embedding_model.device 


    # --- Caching Methods ---
    def get_cache_path(self, identifier: str) -> str:
        """Generates a unique path for cache files based on an identifier."""
        # Ensure identifier is safe for filenames
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
            if not cached_at_str: return False
            
            cached_date = datetime.fromisoformat(cached_at_str)
            return datetime.now() - cached_date < timedelta(days=self.cache_duration_days)
        except (json.JSONDecodeError, FileNotFoundError, KeyError, ValueError) as e:
            st.warning(f"Cache metadata corrupted or invalid for {cache_path}: {e}")
            return False

    def save_to_cache(self, identifier: str) -> None:
        """Saves processed data (chunks and embeddings) to cache files."""
        cache_path = self.get_cache_path(identifier)
        
        try:
            # Save metadata
            metadata = {
                'identifier': identifier,
                'cached_at': datetime.now().isoformat(),
                'files_count': len(self.pdf_contents),
                'chunks_count': len(self.text_chunks)
            }
            with open(f"{cache_path}_meta.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f)
                
            # Save text chunks
            with open(f"{cache_path}_chunks.json", 'w', encoding='utf-8') as f:
                json.dump(self.text_chunks, f)
                
            # Save embeddings
            if self.chunk_embeddings is not None:
                # Ensure embeddings are numpy array before saving, moving to CPU if necessary
                np.save(f"{cache_path}_embeddings.npy", self.chunk_embeddings.cpu().numpy())
            st.success(f"Cache saved for identifier: {identifier}")
        except Exception as e:
            st.error(f"Error saving to cache for {identifier}: {e}")

    def load_from_cache(self, identifier: str) -> bool:
        """Loads data from cache if it exists and is valid."""
        cache_path = self.get_cache_path(identifier)
        if not self.is_cache_valid(cache_path):
            return False
            
        try:
            with open(f"{cache_path}_chunks.json", 'r', encoding='utf-8') as f:
                self.text_chunks = json.load(f)
            
            # Load embeddings and convert back to tensor, moving to the model's device
            loaded_embeddings = np.load(f"{cache_path}_embeddings.npy")
            # Move the loaded tensor to the same device as the embedding model
            self.chunk_embeddings = torch.from_numpy(loaded_embeddings).to(self.model_device)

            # We don't need to load pdf_contents as it's not used after processing
            st.success(f"Cache loaded for identifier: {identifier}")
            return True
        except (FileNotFoundError, json.JSONDecodeError, ValueError, Exception) as e:
            st.warning(f"Could not load from cache: {e}. Reprocessing will be required.")
            return False

    def get_cache_info(self, identifier: str) -> Dict:
        """Retrieves information about the cache for a given identifier."""
        cache_path = self.get_cache_path(identifier)
        meta_path = f"{cache_path}_meta.json"
        
        if not os.path.exists(meta_path):
            return {'exists': False}
        
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            cached_at_str = metadata.get('cached_at', '')
            if not cached_at_str:
                return {'exists': True, 'is_valid': False, 'message': 'Cached date missing'}

            cached_date = datetime.fromisoformat(cached_at_str)
            days_old = (datetime.now() - cached_date).days
            
            return {
                'exists': True,
                'is_valid': days_old < self.cache_duration_days,
                'days_old': days_old,
                **metadata
            }
        except Exception as e:
            st.warning(f"Error reading cache info for {identifier}: {e}")
            return {'exists': False, 'message': f"Error reading cache info: {e}"}

    # --- PDF Processing and Chunking ---
    def extract_text_from_pdf(self, pdf_path: str, filename: str) -> None:
        """Extracts text from a single PDF and stores it."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text() + "\n" # Add newline between pages for better separation
            doc.close()
            
            # Basic text cleaning and normalization
            text = re.sub(r'\s+', ' ', text).strip() # Replace multiple whitespaces with single space
            text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text) # Handle hyphenated words across lines

            if len(text) > 100:  # Only add if content is substantial
                self.pdf_contents[filename] = text
                st.success(f"✅ Extracted {len(text):,} characters from {filename}")
            else:
                st.warning(f"⚠️ Could not extract meaningful content from {filename}.")
        except Exception as e:
            st.error(f"❌ Error processing {filename}: {e}")

    def process_uploaded_files(self, uploaded_files) -> None:
        """Processes a list of uploaded PDF files."""
        self.pdf_contents = {} # Reset for new uploads
        if not uploaded_files:
            st.warning("No files uploaded to process.")
            return

        progress_text = "Processing uploaded files..."
        progress_bar = st.progress(0, text=progress_text)
        
        for i, uploaded_file in enumerate(uploaded_files):
            temp_path = os.path.join(self.uploads_dir, uploaded_file.name)
            try:
                # Save uploaded file to a temporary location
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                self.extract_text_from_pdf(temp_path, uploaded_file.name)
            except Exception as e:
                st.error(f"Failed to save or process {uploaded_file.name}: {e}")
            finally:
                # Ensure temporary file is removed
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
        
        progress_bar.empty()
        if not self.pdf_contents:
            st.error("No content was successfully extracted from any of the PDFs.")


    def recursive_chunk_text(self, text: str, source: str, max_chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict]:
        """
        Recursively splits text into chunks with overlap, prioritizing sentence and paragraph boundaries.
        A more robust way to chunk text to maintain context.
        """
        # Prioritize splitting by paragraph, then sentence, then words
        separators = ["\n\n", "\n", r"(?<=\.|\?|\!)\s+", " ", ""] # Added regex for sentence split
        
        chunks = []
        current_text = text
        
        def _split_recursively(segment: str, current_separator_index: int):
            if not segment:
                return
            
            if len(segment) <= max_chunk_size:
                chunks.append(segment)
                return

            if current_separator_index >= len(separators):
                # Fallback to simple max_chunk_size split if no natural separator works
                for i in range(0, len(segment), max_chunk_size - chunk_overlap):
                    chunks.append(segment[i:i + max_chunk_size])
                return

            sep = separators[current_separator_index]
            if not sep: # Handle empty separator (character by character)
                _split_recursively(segment, current_separator_index + 1)
                return

            parts = re.split(sep, segment) if sep else list(segment) # Split by char if sep is empty
            
            temp_chunk = []
            current_chunk_length = 0
            
            for part in parts:
                if sep and part: # Add separator back if not splitting by char
                    part_with_sep = part + (sep if sep != " " else " ") # Add separator back appropriately
                else:
                    part_with_sep = part

                if current_chunk_length + len(part_with_sep) > max_chunk_size:
                    if temp_chunk:
                        _split_recursively("".join(temp_chunk).strip(), current_separator_index + 1)
                    temp_chunk = [part]
                    current_chunk_length = len(part)
                else:
                    temp_chunk.append(part_with_sep)
                    current_chunk_length += len(part_with_sep)
            
            if temp_chunk:
                _split_recursively("".join(temp_chunk).strip(), current_separator_index + 1)
                
        _split_recursively(text, 0)
        
        # Now apply overlap
        overlapping_chunks = []
        for i in range(len(chunks)):
            chunk_content = chunks[i].strip()
            if not chunk_content:
                continue

            # Add overlap from the end of the previous chunk
            if i > 0 and chunk_overlap > 0:
                previous_chunk_end = chunks[i-1].strip()[-chunk_overlap:]
                chunk_content = previous_chunk_end + " " + chunk_content # Add space to separate overlap
            
            overlapping_chunks.append({'text': chunk_content, 'source': source})
            
        return overlapping_chunks

    def create_chunks_and_embeddings(self) -> None:
        """Processes all PDF content into chunks and generates embeddings."""
        self.text_chunks = []
        if not self.pdf_contents:
            st.error("No PDF content available to chunk. Please upload and process PDFs first.")
            return

        for filename, content in self.pdf_contents.items():
            self.text_chunks.extend(self.recursive_chunk_text(content, filename))
        
        if not self.text_chunks:
            st.error("No text could be chunked from the PDFs after processing.")
            return

        # Generate embeddings
        st.info(f"Generating embeddings for {len(self.text_chunks)} text chunks...")
        chunk_texts = [chunk['text'] for chunk in self.text_chunks]
        try:
            # Added convert_to_tensor=True to ensure compatibility with util.cos_sim
            self.chunk_embeddings = self.embedding_model.encode(chunk_texts, convert_to_tensor=True, show_progress_bar=True)
            st.success("Embeddings generated successfully!")
        except Exception as e:
            st.error(f"Error generating embeddings: {e}. Check if the embedding model loaded correctly or if input texts are too large.")
            self.chunk_embeddings = None # Reset embeddings on failure

    # --- Answering ---
    def find_relevant_chunks(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Finds the most relevant chunks using semantic search (vector similarity).
        """
        if self.chunk_embeddings is None or len(self.text_chunks) == 0:
            st.warning("No embeddings or text chunks available to search. Please process PDFs first.")
            return []
            
        try:
            question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
            
            # Calculate cosine similarity
            # Ensure both are on the same device (CPU/GPU) if applicable
            cosine_scores = util.cos_sim(question_embedding, self.chunk_embeddings)[0]
            
            # Get top_k scores and their indices. Use .cpu().numpy() to convert tensor to numpy for argpartition
            top_results_indices = np.argpartition(-cosine_scores.cpu().numpy(), range(top_k))[:top_k]
            
            relevant_chunks = []
            for idx in top_results_indices:
                chunk = self.text_chunks[idx]
                chunk['score'] = cosine_scores[idx].item() # .item() to get scalar from tensor
                relevant_chunks.append(chunk)
                
            # Sort by score in descending order
            relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
            return relevant_chunks
        except Exception as e:
            st.error(f"Error finding relevant chunks: {e}. Ensure embeddings are correctly generated.")
            return []

    def generate_answer(self, question: str, context_chunks: List[Dict], client: Any) -> str:
        """Generates an answer using the Hugging Face Inference API based on the provided context."""
        if not context_chunks:
            return "I couldn't find relevant information in the PDF files to answer your question."
        
        context_str = ""
        sources = set()
        for i, chunk in enumerate(context_chunks):
            context_str += f"--- Context Chunk {i+1} (Source: {chunk['source']}) ---\n"
            context_str += chunk['text'] + "\n\n"
            sources.add(chunk['source'])

        # Updated prompt format for instruction-tuned models like Mistral-7B-Instruct
        # Using a simple chat template for direct question answering
        full_prompt = f"""[INST] You are a helpful and precise AI assistant that answers questions based *only* on the provided context from PDF documents.
Follow the instructions strictly.

Here is the relevant context from the uploaded documents:
{context_str}

---

Based *only* on the context provided above, please answer the following question.

Question: {question}

Instructions for your answer:
1.  Synthesize a comprehensive and accurate answer from all the provided context chunks.
2.  **Crucially, do not use any information outside of the provided context.**
3.  If the context does not contain the answer, or enough information to answer the question, clearly state: "Based on the provided documents, I cannot find enough information to answer your question."
4.  **Cite the source document (e.g., "[Source: document_name.pdf]") directly after the information you use from it.** If information comes from multiple sources, cite all relevant sources.
5.  Format your answer clearly using paragraphs, bullet points, or numbered lists if helpful for readability.
6.  Maintain a professional, objective, and factual tone. Avoid conversational filler. [/INST]

Answer:"""

        try:
            # Using Hugging Face InferenceClient for conversational task
            # The 'conversational' method takes past_user_inputs, generated_responses, and current_input
            # For a single turn RAG, current_input is sufficient.
            response = client.conversational(
                text=full_prompt, # Send the entire prompt as the current input
                model="mistralai/Mistral-7B-Instruct-v0.2", # Example open-source model
                # Parameters typically go into a 'parameters' dict for conversational
                parameters={
                    "max_new_tokens": 1000, # Max tokens for the generated response
                    "temperature": 0.1,
                    "top_p": 1.0,
                    # "repetition_penalty": 1.05 # Removed as it's not a standard parameter for 'conversational' method
                }
            )
            # The conversational method returns an object with 'generated_text'
            answer = response.generated_text.strip()
            return answer
        except Exception as e:
            return f"Error generating response from Hugging Face: {e}. Please check your Hugging Face API key and model availability."


# --- STREAMLIT UI ---

def get_hf_client():
    """Gets the Hugging Face InferenceClient, prioritizing environment variables over Streamlit secrets."""
    # Use HUGGINGFACE_API_KEY for the token
    api_key = st.secrets.get("HUGGINGFACE_API_KEY") 
    
    if not api_key:
        st.warning("🚨 Hugging Face API key not found! Please set it as a Streamlit secret (in .streamlit/secrets.toml) or an environment variable named `HUGGINGFACE_API_KEY`.")
        return None
    
    try:
        # Initialize the InferenceClient with the API key
        return InferenceClient(token=api_key)
    except Exception as e:
        st.error(f"Error initializing Hugging Face client: {e}. Check if your API key is valid.")
        return None

def main():
    st.title("🤖 Advanced PDF Chatbot")
    st.markdown("A chatbot with Semantic Search, Better Chunking, and Source Attribution, powered by your PDFs.")

    # Initialize session state variables if they don't exist
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
        st.session_state.messages = []
        st.session_state.pdfs_processed = False
        st.session_state.last_used_identifier = None

    # Get the Hugging Face client instead of OpenAI
    hf_client = get_hf_client()
    if hf_client is None:
        st.stop() # Stop execution if no valid Hugging Face client can be initialized
        
    # --- Sidebar for PDF Upload and Processing ---
    with st.sidebar:
        st.header("📄 PDF Configuration")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDFs for the chatbot to analyze. Large files may take longer."
        )

        identifier = ""
        if uploaded_files:
            # Create a stable identifier based on file names and sizes for caching
            file_details = [f"{f.name}-{f.size}" for f in sorted(uploaded_files, key=lambda x: x.name)]
            identifier = "uploaded_" + hashlib.sha256("|".join(file_details).encode('utf-8')).hexdigest()

        cache_info_display = {}
        if identifier:
            cache_info_display = st.session_state.chatbot.get_cache_info(identifier)
            if cache_info_display.get('exists'):
                status = "✅ Valid" if cache_info_display['is_valid'] else "⚠️ Expired"
                st.info(f"Cache Status: {status} ({cache_info_display['days_old']} days old)")
                with st.expander("Cache Details"):
                    st.json(cache_info_display)
            else:
                st.info("No cache found for these files. Processing will be required.")

        process_button_disabled = not uploaded_files
        if st.session_state.last_used_identifier == identifier and cache_info_display.get('is_valid', False) and st.session_state.pdfs_processed:
            st.success("PDFs already processed and loaded from valid cache.")
            process_button_disabled = True
            
        if st.button("🔄 Process PDFs", disabled=process_button_disabled):
            st.session_state.pdfs_processed = False
            st.session_state.messages = [] # Clear chat on new processing to avoid stale context
            st.session_state.last_used_identifier = identifier # Update last used identifier
            
            with st.spinner("Processing documents and generating embeddings... This may take a moment."):
                if st.session_state.chatbot.load_from_cache(identifier):
                    st.session_state.pdfs_processed = True
                else:
                    st.session_state.chatbot.process_uploaded_files(uploaded_files)
                    if st.session_state.chatbot.pdf_contents:
                        st.session_state.chatbot.create_chunks_and_embeddings()
                        if st.session_state.chatbot.chunk_embeddings is not None: # Only save if embeddings were successful
                            st.session_state.chatbot.save_to_cache(identifier)
                            st.session_state.pdfs_processed = True
                        else:
                            st.error("Embedding generation failed. Please check PDF content and try again.")
                    else:
                        st.error("No content could be extracted from the provided PDFs for chunking.")
            
            if st.session_state.pdfs_processed:
                st.success("✅ PDFs processed and ready!")
            else:
                st.error("Failed to process PDFs. Please check for errors above.")

    # --- Main Chat Interface ---
    if not st.session_state.pdfs_processed:
        st.info("👈 Please upload your PDF documents and click 'Process PDFs' to begin chatting.")
    else:
        st.header("💬 Ask a Question about your PDFs")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "context" in message and st.session_state.get('show_context', False): # Only show context if toggled
                    with st.expander("Retrieved Context"):
                        if message["context"]:
                            for i, chunk in enumerate(message["context"]):
                                st.info(f"Chunk {i+1} (Source: {chunk.get('source', 'N/A')}, Score: {chunk.get('score', 0):.4f})")
                                st.text(chunk['text'])
                        else:
                            st.warning("No relevant chunks were found for this query.")
        
        # Toggle to show/hide context
        with st.sidebar:
            st.session_state.show_context = st.checkbox("Show Retrieved Context in Chat", value=False)


        # User input
        if question := st.chat_input("Ask about the content of the PDFs..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
                
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                with st.spinner("Searching and generating answer..."):
                    relevant_chunks = st.session_state.chatbot.find_relevant_chunks(question)
                    # Pass the Hugging Face client
                    answer = st.session_state.chatbot.generate_answer(question, relevant_chunks, hf_client)
                    
                    full_response = answer
                    message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response, 
                "context": relevant_chunks # Store context for potential display
            })
            
    # Clear chat button in sidebar
    if st.session_state.messages:
        with st.sidebar:
            st.markdown("---")
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

if __name__ == "__main__":
    main()
