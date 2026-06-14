

import streamlit as st
import tempfile
import chromadb
import ollama

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Multi-PDF Research Assistant")

st.title("📚 Multi-PDF Research Assistant")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_documents = []

    with st.spinner("Loading PDFs..."):

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

    st.success(
        f"{len(uploaded_files)} PDFs Loaded Successfully!"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        all_documents
    )

    st.write(
        f"Total Chunks: {len(chunks)}"
    )

    embed_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    with st.spinner("Creating Embeddings..."):

        embeddings = embed_model.encode(
            texts
        ).tolist()

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    try:
        client.delete_collection(
            "research_papers"
        )
    except:
        pass

    collection = client.create_collection(
        "research_papers"
    )

    batch_size = 100

    for i in range(
        0,
        len(chunks),
        batch_size
    ):

        docs_batch = [
            chunk.page_content
            for chunk in chunks[i:i+batch_size]
        ]

        emb_batch = embeddings[i:i+batch_size]

        ids_batch = [
            str(x)
            for x in range(
                i,
                i + len(docs_batch)
            )
        ]

        collection.add(
            ids=ids_batch,
            documents=docs_batch,
            embeddings=emb_batch
        )

    st.success(
        "Knowledge Base Created!"
    )

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        query_embedding = embed_model.encode(
            [question]
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=4
        )

        context = "\n".join(
            results["documents"][0]
        )

        prompt = f"""
You are a research assistant.

Answer only from the provided context.

If the answer is not present,
say:
"I could not find the answer in the uploaded PDFs."

Context:
{context}

Question:
{question}
"""

        with st.spinner("Thinking..."):

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "system",
                        "content":
                        "Answer only from context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

        st.subheader("Answer")

        st.write(
            response["message"]["content"]
        )