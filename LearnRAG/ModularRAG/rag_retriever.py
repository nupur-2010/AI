from typing import List, Dict, Any
from embeddings_generator import EmbeddingsGenerator
from vector_store import VectorStore

class RAGretriever:
    def __init__(self, embeddings_generator : EmbeddingsGenerator, vector_store : VectorStore):
        self.embeddings_generator = embeddings_generator
        self.vector_store = vector_store

    def retrieve_results(self, query : str, top_k : int, score_threshold : float) -> List[Dict[str,Any]]:
        print(f"Retrieving top {top_k} results with score greater than {score_threshold} for query: {query}")
        print()
        query_embedding = self.embeddings_generator.generate_embedding([query])[0]
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            retrieved_results = []
            if results["documents"] and results["documents"][0]:
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                for i, (id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_results.append({
                            "id" : id,
                            "page_content" : document,
                            "metadata" : metadata,
                            "similarity_score" : similarity_score,
                            "distance" : distance,
                            "rank" : i + 1
                        })
                print(f"Retrived {len(retrieved_results)} similar results for the query - {query}")
                print()
            else:
                print(f"No results found similar to the query - {query}")
                print()
            return retrieved_results
        except Exception as e:
            print(f"Error occured while retrieving results similar to the query - {query} : {e}")
            print()