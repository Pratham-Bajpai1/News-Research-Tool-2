from typing import Optional
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from sentence_transformers import CrossEncoder
import streamlit as st
import re

def setup_qa_system():
    """Initialize QA system components"""
    return {
        'llm': HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.3,
            max_length=1024,
            top_p=0.9,
            top_k=50
        ),
        'cross_encoder': CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    }


def get_answer(query: str, faiss_index, qa_system: dict) -> str:
    """Core QA logic"""
    retriever = faiss_index.as_retriever(search_kwargs={"k": 5})

    # Rerank results if cross-encoder is available
    if qa_system['cross_encoder']:
        results = retriever.get_relevant_documents(query)
        pairs = [[query, doc.page_content] for doc in results]
        scores = qa_system['cross_encoder'].predict(pairs)
        results = [doc for _, doc in sorted(zip(scores, results), reverse=True)]

    qa_chain = RetrievalQA.from_chain_type(
        llm=qa_system['llm'],
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    response = qa_chain({"query": query})
    answer = response["result"].split("Helpful Answer:")[-1].strip()

    # Special handling for price differences
    if "difference in price" in query.lower():
        prices = re.findall(r'[\d.]+', answer)
        if len(prices) >= 2:
            difference = float(prices[1]) - float(prices[0])
            answer = f"The price difference is: {abs(difference):,.2f}"

    return answer, response.get("source_documents")


def show_qa_ui(faiss_index, input_lang, output_lang, translator):
    """Modern multilingual QA interface"""
    qa_system = setup_qa_system()

    st.subheader("❓ Ask Questions")
    query = st.text_input(
        "Enter your question about the articles",
        placeholder="What are the key points about...?"
    )

    if query:
        with st.spinner("Analyzing articles..."):
            try:
                # Translate query to English if needed
                working_query = query
                if input_lang != "eng_Latn":
                    working_query = translator.translate_text(query, "eng_Latn", input_lang)

                # Get answer
                answer, sources = get_answer(working_query, faiss_index, qa_system)

                # Translate answer back if needed
                if output_lang != "eng_Latn":
                    answer = translator.translate_text(answer, output_lang, "eng_Latn")

                st.markdown("### 💡 Answer")
                st.markdown(f'<div class="custom-card">{answer}</div>', unsafe_allow_html=True)

                if sources:
                    st.markdown("### 📚 Sources")
                    for doc in sources:
                        st.markdown(f"""
                        <div class="custom-card">
                            <p><strong>Source:</strong> <a href="{doc.metadata['source']}" target="_blank">{doc.metadata['source']}</a></p>
                            <p>{doc.page_content[:300]}...</p>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")