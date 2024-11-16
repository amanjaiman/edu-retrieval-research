import time

from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from helpers import batch_convert_to_graph_documents, extract_documents_from_pdfs

def main(dir_path, chunk_size):
    start_time = time.time()

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    llm_transformer = LLMGraphTransformer(llm=llm)
    checkpoint_1 = time.time()
    print(f"Initialized LLM. Total time: {checkpoint_1 - start_time}")

    graph = Neo4jGraph()
    checkpoint_2 = time.time()
    print(f"Connected to Neo4j instance. Total time: {checkpoint_2 - start_time}")

    documents = extract_documents_from_pdfs(dir_path, chunk_size) # ./BiochemData, 512
    checkpoint_3 = time.time()
    print(f"Created {len(documents)} documents from data. Total time: {checkpoint_3 - start_time}")

    graph_documents = batch_convert_to_graph_documents(llm_transformer, documents)
    checkpoint_4 = time.time()
    print(f"Graph Documents created. Total time: {checkpoint_4 - start_time}")

    graph.add_graph_documents(graph_documents)
    checkpoint_5 = time.time()
    print(f"Graph creation complete. Total time: {checkpoint_5 - start_time}")