# Key modifications to add to your simple_askscottie_iframes_test.py file

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

# --- IFRAME-SPECIFIC CONFIGURATION ---
st.set_page_config(
    page_title="Ask Scottie - Maryville College Academic Assistant",
    page_icon="🐕‍🦺",
    layout="wide",
    initial_sidebar_state="collapsed"  # Changed to collapsed for iframe
)

# Add iframe-specific parameter handling
query_params = st.query_params
is_iframe = query_params.get("iframe", "false").lower() == "true"

# Enhanced CSS for iframe mode with better responsive design
if is_iframe:
    st.markdown("""
    <style>
        /* Hide Streamlit branding and unnecessary elements in iframe mode */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Adjust padding for iframe */
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        
        /* Hide sidebar toggle in iframe mode */
        button[kind="header"] {
            display: none !important;
        }
        
        /* Responsive design for iframe */
        .stChatFloatingInputContainer {
            bottom: 0.5rem !important;
        }
        
        /* Compact header for iframe */
        .main-header {
            background: linear-gradient(135deg, #5B0F1B, #EC5E1A);
            color: white;
            padding: 10px !important;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 10px;
        }
        
        .main-header h1 {
            font-size: 1.5em !important;
            margin: 0 !important;
        }
        
        .main-header p {
            font-size: 0.9em !important;
            margin: 3px 0 0 0 !important;
        }
        
        /* Adjust info box for iframe */
        .info-box {
            background: #E3F2FD;
            border-left: 4px solid #1976D2;
            padding: 10px;
            border-radius: 5px;
            margin: 8px 0;
            font-size: 0.9em;
        }
        
        /* Optimize chat messages for iframe */
        .stChatMessage {
            padding: 0.5rem !important;
        }
        
        /* Source items styling */
        .source-item {
            background: #F5F5F5;
            border: 1px solid #DDD;
            padding: 8px;
            margin: 4px 0;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        /* Make sample questions more compact */
        .stButton > button {
            font-size: 0.85em !important;
            padding: 0.4rem 0.8rem !important;
        }
        
        /* Ensure proper scrolling in iframe */
        .main {
            overflow-y: auto !important;
            height: 100vh !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    # Original styling for standalone mode
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

# [Include all your existing Config and InstitutionalPDFChatbot classes here - no changes needed]
# ... (rest of your existing code remains the same)

def main():
    # Conditional header based on iframe mode
    if is_iframe:
        # Compact header for iframe
        st.markdown("""
        <div class="main-header">
            <h1>🐕‍🦺 Ask Scottie</h1>
            <p>Maryville College Academic Catalog Assistant</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Full header for standalone
        st.markdown("""
        <div class="main-header">
            <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
                <img src="data:image/png;base64,iVBORw0" 
                     alt="Scottie Logo" 
                     style="width: 60px; height: 60px; border-radius: 50%; background: white; padding: 5px;">
                <div>
                    <h1 style="margin: 0; font-size: 2.5em;">Ask Scottie</h1>
                    <p style="margin: 5px 0 0 0;">Scottish Terrier Academic Assistant • Maryville College Academic Catalog Helper</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Scope limitation notice (make it smaller in iframe mode)
    if not is_iframe or len(st.session_state.get('messages', [])) == 0:
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
        
        if not is_iframe:  # Only show detailed status in standalone mode
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
    
    # Only show sidebar in standalone mode
    if not is_iframe:
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
        # In iframe mode, show a more compact initial view
        if is_iframe:
            st.markdown("### 💡 Try asking about:")
            sample_questions = [
                "General education requirements",
                "Grading policy",
                "Admission requirements",
                "Graduation requirements"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(sample_questions):
                with cols[i % 2]:
                    if st.button(question, key=f"sample_{i}", use_container_width=True):
                        full_question = f"What are the {question.lower()}?" if not question.startswith("What") else question
                        st.session_state.messages.append({"role": "user", "content": full_question})
                        
                        with st.spinner("🔍 Searching..."):
                            relevant_chunks = chatbot.context_aware_search(
                                full_question, 
                                conversation_history=st.session_state.messages
                            )
                            
                            answer = chatbot.generate_answer(
                                full_question, 
                                relevant_chunks, 
                                openai_client,
                                conversation_history=st.session_state.messages
                            )
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.rerun()
        else:
            # Original detailed view for standalone mode
            # [Keep your existing code for showing available sources and sample questions]
            pass  # Your existing code here
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "system":
            st.markdown(f'<div style="text-align: center; color: #666; margin: 10px 0; font-style: italic;">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask about the Academic Catalog..."):
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