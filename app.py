import streamlit as st
import tempfile
import numpy as np
import faiss

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

st.title("📚 Multi PDF Research Assistant")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_documents = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        all_documents.extend(documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_documents)

    st.success(f"Created {len(chunks)} chunks")

    texts = [chunk.page_content for chunk in chunks]

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        query_embedding = model.encode(
            [question]
        )

        distances, indices = index.search(
            np.array(query_embedding).astype("float32"),
            k=3
        )

        st.subheader(
            "Top Matching Chunks"
        )

        for idx in indices[0]:

            st.write(texts[idx])
            st.divider()
