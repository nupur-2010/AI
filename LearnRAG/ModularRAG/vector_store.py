import os
import chromadb
import uuid
import numpy as np
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, collection_name : str, persist_directory : str):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description" : "PDF Document embeddings for RAG"}
            )
            print(f"Vector store initialized.")
            print()
            print(f"Existing collections in the vector store {self.collection_name}: {self.collection.count()}")
            print()
        except Exception as e:
            print(f"Error occured while initializing vector store {self.collection_name} : {e}")
            print()

    def add_documents_embeddings(self, documents : list[Any], embeddings : np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("The number of documents must match the number of embeddings.")
        print(f"Adding {len(documents)} records to the vector store...")
        print()
        id_list = []
        page_content_list = []
        metadata_list = []
        embedding_list = []
        for i, (document, embedding) in enumerate(zip(documents,embeddings)):
            id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            id_list.append(id)
            page_content_list.append(document.page_content)
            metadata = dict(document.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(document.page_content)
            metadata_list.append(metadata)
            embedding_list.append(embedding.tolist())
        try:
            self.collection.add(
                ids=id_list,
                documents=page_content_list,
                metadatas=metadata_list,
                embeddings=embedding_list
            )
            print(f"Successfully added {len(documents)} records to the vector store.")
            print()
            print(f"Existing collections in the vector store {self.collection_name}: {self.collection.count()}")
            print()
        except Exception as e:
            print(f"Error occured while adding records to the vector store")
            print()