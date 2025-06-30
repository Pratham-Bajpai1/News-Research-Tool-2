import os
from translation import show_language_selector, Translator
from typing import List, Optional

import streamlit as st
import pickle
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from summarization import show_summarization_ui
from qa_system import setup_qa_system, show_qa_ui  # We'll create this next

from feedback import show_feedback_page


# Initialize app
load_dotenv()


# Custom CSS injection
def inject_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Dark/light mode detection
    st.markdown("""
        <script>
        const mode = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        document.body.setAttribute('data-theme', mode);
        </script>
        """, unsafe_allow_html=True)


# New modern header
def show_header():
    st.markdown("""
    <div style="padding: 2rem 0; border-bottom: 1px solid var(--accent); margin-bottom: 2rem;">
        <h1 style="color: var(--primary); margin-bottom: 0.5rem;">📰 News Research Pro</h1>
        <p style="color: var(--secondary-text); margin-top: 0;">Advanced AI-powered news analysis tool</p>
        </div>
    """, unsafe_allow_html=True)


# Initialize app with new config
st.set_page_config(
    page_title="News Research Pro",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()
show_header()

# --- Constants ---
MAX_URLS = 3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Session State ---
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None


# --- Helper Functions ---
def clean_html(content: str) -> str:
    """Remove unnecessary HTML elements and clean text"""
    soup = BeautifulSoup(content, 'html.parser')
    for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'header', 'button', 'form']):
        element.decompose()
    return ' '.join(soup.get_text(separator=' ', strip=True).split())


async def fetch_url(session: aiohttp.ClientSession, url: str) -> Optional[Document]:
    """Fetch and clean single URL with robust error handling"""
    try:
        async with session.get(url, timeout=15) as response:
            response.raise_for_status()
            text = await response.text()
            cleaned_content = clean_html(text)
            if len(cleaned_content) < 200:
                st.warning(f"Content too short from {url}")
                return None
            return Document(page_content=cleaned_content, metadata={"source": url})
    except Exception as e:
        st.error(f"Error fetching {url}: {str(e)}")
        return None


async def fetch_all_urls(urls: List[str]) -> List[Document]:
    """Fetch multiple URLs concurrently with progress"""
    with st.spinner(f"Fetching {len(urls)} URLs..."):
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_url(session, url) for url in urls if url and url.startswith(('http://', 'https://'))]
            results = await asyncio.gather(*tasks)
            return [doc for doc in results if doc is not None]


def process_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks with optimal settings"""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    return text_splitter.split_documents(docs)


# --- New Sidebar Design ---
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0; border-bottom: 1px solid var(--accent); margin-bottom: 1.5rem;">
        <h2 style="color: var(--primary); margin-bottom: 0;">Configuration</h2>
    </div>
    """, unsafe_allow_html=True)

    # URL Processing with new design
    st.markdown("### 📌 Article URLs")
    urls = [st.text_input(f"URL {i + 1}", key=f"url_{i}",
                          placeholder="https://example.com/news-article")
            for i in range(MAX_URLS)]

    if st.button("🔄 Process Articles", use_container_width=True):
        if not any(urls):
            st.error("Please enter at least one valid URL")
        else:
            with st.spinner("Processing articles..."):
                fetched_docs = asyncio.run(fetch_all_urls(urls))
                if fetched_docs:
                    processed_docs = process_documents(fetched_docs)
                    st.session_state.faiss_index = FAISS.from_documents(
                        processed_docs,
                        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                    )
                    st.session_state.processed_docs = fetched_docs
                    st.success(f"✅ Processed {len(fetched_docs)} articles")
                else:
                    st.error("Failed to process any URLs")

    # Navigation with icons
    if st.session_state.processed_docs:
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Go to",
            ["Main Features", "Feedback"],
            index=0,
            label_visibility="collapsed"
        )

        if page == "Feedback":
            st.session_state.current_page = "feedback"
        else:
            st.session_state.current_page = "main"
    else:
        st.session_state.current_page = "main"

# --- Main Content ---
if st.session_state.get('current_page') == "feedback":
    show_feedback_page()
    st.stop()

if st.session_state.processed_docs:
    # Feature selection with tabs
    st.markdown("### 🔍 Analysis Tools")
    feature_tabs = st.tabs(["📝 Summarization", "❓ Q&A", "🌐 Translation"])

    # Add language selection for all features
    input_lang, output_lang = show_language_selector()
    translator = Translator()

    with feature_tabs[0]:  # Summarization
        show_summarization_ui(
            st.session_state.processed_docs,
            input_lang,
            output_lang,
            translator
        )

    with feature_tabs[1]:  # Question Answering
        show_qa_ui(
            st.session_state.faiss_index,
            input_lang,
            output_lang,
            translator
        )

    with feature_tabs[2]:  # Translation
        st.subheader("Document Translation")
        if st.button("Translate Documents", key="translate_btn"):
            with st.spinner(f"Translating from {input_lang} to {output_lang}..."):
                translated_docs = [
                    translator.translate_document(doc, output_lang)
                    for doc in st.session_state.processed_docs
                ]
                st.session_state.translated_docs = translated_docs
                st.success("Translation complete!")

                for i, doc in enumerate(translated_docs):
                    with st.expander(f"Translated Document {i + 1}"):
                        st.write(doc.page_content[:2000] + "...")

# --- Premium Footer ---
st.markdown("---")

# First row with tips and version
st.markdown("""
<div class="footer-top">
    <div class="footer-tips">
        <strong>Tips:</strong> Use complete URLs  •  300+ word articles work best
    </div>
    <div class="footer-version">
        News Research Pro • v1.0
    </div>
</div>

<div class="footer-bottom">
    <div class="copyright">
        © 2025 Crafted with <span style="color: var(--primary);">♥</span> by Pratham Bajpai
    </div>
</div>
""", unsafe_allow_html=True)