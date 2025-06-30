from transformers import pipeline
from typing import Optional, Union
import streamlit as st
from langchain.schema import Document


class Translator:
    def __init__(self):
        '''
        # In translation.py constructor For production use:
        self.model = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-1.3B",  # Larger model
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        '''
        # Using Facebook's NLLB model for translation
        self.model = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            src_lang="eng_Latn",  # Default source (English)
            device="cpu"  # Use "cuda" if GPU available
        )

    def detect_language(self, text: str) -> str:
        """Simple language detection (for demo - replace with proper detection if needed)"""
        if any(char in text for char in ['。', '的', '是']):
            return "zho_Hans"  # Chinese
        elif any(char in text for char in ['の', 'です', 'ます']):
            return "jpn_Jpan"  # Japanese
        return "eng_Latn"  # Default to English

    def translate_text(
            self,
            text: str,
            target_lang: str = "eng_Latn",
            source_lang: Optional[str] = None
    ) -> str:
        """Translate text between languages"""
        try:
            if not text.strip():
                return text

            if source_lang is None:
                source_lang = self.detect_language(text)

            if source_lang == target_lang:
                return text

            result = self.model(
                text,
                src_lang=source_lang,
                tgt_lang=target_lang,
                max_length=400
            )
            return result[0]['translation_text']
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return text

    def translate_document(self, doc: Document, target_lang: str) -> Document:
        """Translate a Langchain document"""
        translated_content = self.translate_text(doc.page_content, target_lang)
        return Document(
            page_content=translated_content,
            metadata=doc.metadata
        )


# Supported languages (NLLB language codes)
SUPPORTED_LANGUAGES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Arabic": "arb_Arab"
}


def show_language_selector() -> tuple:
    """Display language selection UI"""
    col1, col2 = st.columns(2)
    with col1:
        input_lang = st.selectbox(
            "Input Language",
            list(SUPPORTED_LANGUAGES.keys()),
            index=0
        )
    with col2:
        output_lang = st.selectbox(
            "Output Language",
            list(SUPPORTED_LANGUAGES.keys()),
            index=0
        )
    return SUPPORTED_LANGUAGES[input_lang], SUPPORTED_LANGUAGES[output_lang]