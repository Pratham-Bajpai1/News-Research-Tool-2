from typing import List, Optional, Tuple
from langchain.schema import Document
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import streamlit as st


class Summarizer:
    def __init__(self):
        self.model = pipeline("summarization", model="facebook/bart-large-cnn")
        self.min_length = 100  # Minimum characters required for summarization

    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """Generate summary with robust error handling"""
        try:
            if len(text) < self.min_length:
                return "Content too short for meaningful summary"

            if len(text) > 10000:
                text = text[:10000] + "... [truncated]"

            summary = self.model(
                text,
                max_length=max_length,
                min_length=50,
                do_sample=False,
                truncation=True
            )
            return summary[0]["summary_text"]
        except IndexError:
            return "Summary unavailable (content format issue)"
        except Exception as e:
            return f"Summarization error: {str(e)}"

    def generate_individual_summaries(self, docs: List[Document]) -> List[Tuple[str, str]]:
        """Generate summaries for multiple documents in parallel"""
        with ThreadPoolExecutor() as executor:
            return list(executor.map(
                lambda doc: (doc.metadata["source"], self.generate_summary(doc.page_content)),
                docs
            ))

    def generate_combined_summary(self, docs: List[Document]) -> str:
        """Generate a comprehensive summary of multiple documents"""
        # First generate individual summaries
        section_summaries = self.generate_individual_summaries(docs)

        # Then summarize the summaries
        combined_content = "\n".join([f"ARTICLE {i + 1}:\n{summ}"
                                      for i, (_, summ) in enumerate(section_summaries)])
        return self.generate_summary(combined_content, max_length=200)


def show_summarization_ui(docs: List[Document], input_lang: str, output_lang: str, translator):
    """Modern multilingual summarization interface"""
    summarizer = Summarizer()

    # Translate documents if needed
    working_docs = docs
    if input_lang != "eng_Latn":
        with st.spinner("Translating documents for summarization..."):
            working_docs = [
                translator.translate_document(doc, "eng_Latn")
                for doc in docs
            ]

    # Summary type selection
    col1, col2 = st.columns([1, 3])
    with col1:
        summary_type = st.radio(
            "Summary Type",
            ["Individual", "Combined"],
            index=0,
            horizontal=True,
            key="summary_type"
        )

    if summary_type == "Individual":
        st.subheader("📄 Individual Summaries")
        tabs = st.tabs([f"Article {i + 1}" for i in range(len(working_docs))])

        for i, tab in enumerate(tabs):
            with tab:
                doc = working_docs[i]
                source = doc.metadata["source"]

                st.markdown(f"**Source:** [{source}]({source})")
                if st.button(f"Generate Summary", key=f"summarize_{i}"):
                    with st.spinner("Generating summary..."):
                        summary = summarizer.generate_summary(doc.page_content)

                        # Translate back if needed
                        if output_lang != "eng_Latn":
                            summary = translator.translate_text(summary, output_lang, "eng_Latn")

                        st.markdown("### 📋 Summary")
                        st.markdown(f'<div class="custom-card">{summary}</div>', unsafe_allow_html=True)

                        st.markdown("### 🔍 Full Content Preview")
                        st.markdown(f'<div class="custom-card">{doc.page_content[:500]}...</div>',
                                    unsafe_allow_html=True)

    elif summary_type == "Combined":
        st.subheader("📑 Combined Summary")
        if st.button("Generate Combined Summary", key="combined_summary"):
            with st.spinner("Creating comprehensive summary..."):
                # Generate individual summaries first
                summaries = summarizer.generate_individual_summaries(working_docs)
                combined_content = "\n".join([f"ARTICLE {i + 1}:\n{summ}"
                                              for i, (_, summ) in enumerate(summaries)])

                # Generate final summary
                combined_summary = summarizer.generate_summary(combined_content, max_length=200)

                # Translate back if needed
                if output_lang != "eng_Latn":
                    combined_summary = translator.translate_text(combined_summary, output_lang, "eng_Latn")

                st.markdown("### 📜 Comprehensive Summary")
                st.markdown(f'<div class="custom-card">{combined_summary}</div>', unsafe_allow_html=True)

                st.markdown("### 📌 Article Highlights")
                for i, (source, summ) in enumerate(summaries):
                    with st.expander(f"Article {i + 1}: {source}"):
                        if output_lang != "eng_Latn":
                            summ = translator.translate_text(summ, output_lang, "eng_Latn")
                        st.markdown(f'<div class="custom-card">{summ}</div>', unsafe_allow_html=True)