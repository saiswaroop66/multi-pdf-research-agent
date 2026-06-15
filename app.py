import streamlit as st
import tempfile
import numpy as np
import faiss

from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Multi PDF AI Assistant")

st.title("📚 Multi PDF AI Research Assistant")

api_key = st.text_input(
    "Enter Gemini API Key",
    type="password"
)

if api_key:

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:

        all_documents = []

        with st.spinner("Reading PDFs..."):

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

        chunks = splitter.split_documents(
            all_documents
        )

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        st.success(
            f"{len(chunks)} chunks created"
        )

        with st.spinner("Creating Embeddings..."):

            embed_model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

            embeddings = embed_model.encode(
                texts
            )

        embeddings = np.array(
            embeddings
        ).astype("float32")

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(
            dimension
        )

        index.add(
            embeddings
        )

        question = st.text_input(
            "Ask a Question"
        )

        if question:

            query_embedding = embed_model.encode(
                [question]
            )

            query_embedding = np.array(
                query_embedding
            ).astype("float32")

            distances, indices = index.search(
                query_embedding,
                k=4
            )

            context = "\n\n".join(
                [texts[i] for i in indices[0]]
            )

            prompt = f"""
You are a helpful PDF assistant.

Answer ONLY from the context.

If the answer is not found, say:

I could not find the answer in the uploaded PDFs.

Context:
{context}

Question:
{question}
"""

            try:

                client = genai.Client(
                    api_key=api_key
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                st.subheader("Answer")
                st.write(response.text)

            except Exception as e:

                st.error(
                    f"Gemini Error: {e}"
                )
