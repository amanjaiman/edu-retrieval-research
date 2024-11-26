import os
import time
from typing import List, Tuple, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_document import Node, Relationship
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field
from tqdm import tqdm

def extract_documents_from_pdfs(directory, chunk_size):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    documents = []

    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return documents

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)

            try:
                loader = PyPDFLoader(pdf_path)
                raw_documents = loader.load()
                
                try:
                    docs = text_splitter.split_documents(raw_documents)
                    documents += docs
                except TypeError as e:
                    print(f"Error splitting document {filename}: {str(e)}")
                    print(f"Raw document type: {type(raw_documents)}")
                    if raw_documents:
                        print(f"First raw document type: {type(raw_documents[0])}")
                        print(f"First raw document content type: {type(raw_documents[0].page_content)}")
                    
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")

    return documents

def node_to_key(node: Node) -> Tuple[str, str]:
    return (node.id, node.type)

def relationship_to_key(rel: Relationship) -> Tuple[Tuple[str, str], Tuple[str, str], str]:
    return (node_to_key(rel.source), node_to_key(rel.target), rel.type)

def ensure_node_in_tracker(node: Node, node_tracker: Dict[Tuple[str, str], Node]) -> Node:
    key = node_to_key(node)
    if key not in node_tracker:
        node_tracker[key] = node
    return node_tracker[key]

def batch_convert_to_graph_documents(
    llm_transformer,
    documents: List[Document],
    batch_size: int = 10,
    sleep_time: float = 1.0
) -> List[GraphDocument]:
    all_graph_documents = []
    node_tracker: Dict[Tuple[str, str], Node] = {}
    relationship_tracker: Dict[Tuple[Tuple[str, str], Tuple[str, str], str], Relationship] = {}

    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i+batch_size]
        
        # Convert batch to graph documents
        batch_graph_documents = llm_transformer.convert_to_graph_documents(batch)
        
        # Process each graph document in the batch
        for graph_doc in batch_graph_documents:
            new_nodes = []
            new_relationships = []
            
            # Check and deduplicate nodes
            for node in graph_doc.nodes:
                new_nodes.append(ensure_node_in_tracker(node, node_tracker))
            
            # Check and deduplicate relationships
            for rel in graph_doc.relationships:
                rel_key = relationship_to_key(rel)
                if rel_key not in relationship_tracker:
                    # Ensure we're using the deduplicated nodes in the relationship
                    source_node = ensure_node_in_tracker(rel.source, node_tracker)
                    target_node = ensure_node_in_tracker(rel.target, node_tracker)
                    new_rel = Relationship(source=source_node, target=target_node, type=rel.type)
                    relationship_tracker[rel_key] = new_rel
                new_relationships.append(relationship_tracker[rel_key])
            
            # Create a new GraphDocument with deduplicated nodes and relationships
            try:
                graph_doc_args = {
                    "nodes": new_nodes,
                    "relationships": new_relationships,
                }
                if hasattr(graph_doc, 'source'):
                    graph_doc_args['source'] = graph_doc.source
                
                deduplicated_graph_doc = GraphDocument(**graph_doc_args)
                all_graph_documents.append(deduplicated_graph_doc)
            except Exception as e:
                print(f"Error creating GraphDocument: {e}")
                print(f"Nodes: {new_nodes}")
                print(f"Relationships: {new_relationships}")
                if hasattr(graph_doc, 'source'):
                    print(f"Source: {graph_doc.source}")
        
        # Sleep between batches to avoid overwhelming the system
        time.sleep(sleep_time)
    
    return all_graph_documents

# Pydantic
class Answer(BaseModel):
    """The answer to the question"""

    correct_answer: str = Field(description="The single letter answer choice that is correct (A | B | C | D)")
    explanation: str = Field(description="Why the selected answer choice is correct")
    confidence: str = Field(description="Your confidence as a numerical percentage that that is the correct answer (0-100)")