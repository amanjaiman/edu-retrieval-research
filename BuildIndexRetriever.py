import os
from uuid import uuid4

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from helpers import extract_documents_from_pdfs

def main(dir_path, chunk_size):
    embeddings = HuggingFaceEmbeddings()  # model="sentence-transformers/all-mpnet-base-v2"

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    documents = extract_documents_from_pdfs(dir_path, chunk_size) # ./BiochemData, 512

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)

    vector_store.save_local("data_index")