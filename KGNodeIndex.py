import argparse
import faiss
from neo4j import GraphDatabase
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer

class Neo4jGraphNodeLoader:
    def __init__(self):
        uri = os.environ['NEO4J_URI']
        username = os.environ['NEO4J_USERNAME']
        password = os.environ['NEO4J_PASSWORD']
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def get_all_node_names(self):
        query = """
        MATCH (n)
        RETURN n.id AS id, labels(n) AS labels
        """
        results = self.query(query)
        ids = [(record['id'], record['labels'][0]) for record in results]
        return ids

    def close(self):
        self.driver.close()

class WordSimilarityIndex:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.words = []
    
    def load_words(self, words):
        """Load words"""
        self.words = words
    
    def build_index(self):
        """Generate embeddings for each word and build the FAISS index."""
        print(self.words)
        words = [t[0] for t in self.words]
        embeddings = self.model.encode(words)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))
    
    def save_index(self, index_path="kg_node_index.bin", words_path="words.pkl"):
        """Save the FAISS index and words list to disk."""
        if self.index is None:
            raise ValueError("Index is not built. Call `build_index` first.")
        faiss.write_index(self.index, index_path)
        with open(words_path, 'wb') as f:
            pickle.dump(self.words, f)
    
    def load_index(self, index_path="kg_node_index.bin", words_path="words.pkl"):
        """Load the FAISS index and words list from disk."""
        self.index = faiss.read_index(index_path)
        with open(words_path, 'rb') as f:
            self.words = pickle.load(f)
    
    def search_similar_words(self, query, top_k=5):
        """Perform a similarity search for the given query word."""
        if self.index is None:
            raise ValueError("Index is not loaded or built.")
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        results = [self.words[i] for idx, i in enumerate(indices[0])]
        return list(set(results))

def build_index():
    print("Building similarity index on the KG nodes")
    
    graph = Neo4jGraphNodeLoader()
    words = graph.get_all_node_names()

    index = WordSimilarityIndex()
    index.load_words(words)
    index.build_index()

    index.save_index()
    
    print('Done')

# if __name__ == '__main__':
#     # TODO: add arg to specify output path
#     # graph = Neo4jGraphNodeLoader()
#     # words = graph.get_all_node_names()

#     # index = WordSimilarityIndex()
#     # index.load_words(words)
#     # index.build_index()

#     # index.save_index()

#     index = WordSimilarityIndex()
#     index.load_index()

#     results = index.search_similar_words("""Question: Ammonia from brain is removed as:
# A) Urea
# B) Alanine
# C) Glutamate
# D) Glutamine""")
    
#     print(results)
