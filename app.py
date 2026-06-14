import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    st.success(f"{len(uploaded_files)} PDFs Loaded Successfully!")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_documents)

    st.write("Total Pages:", len(all_documents))
    st.write("Total Chunks:", len(chunks))

    st.subheader("Preview Chunks")

    for i, chunk in enumerate(chunks[:5]):
        st.write(f"Chunk {i+1}")
        st.write(chunk.page_content)
        st.divider()
