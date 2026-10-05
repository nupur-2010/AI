import os
from dotenv import load_dotenv
from document_loader import (PDFs_to_Documents,Documents_to_Chunks)
from embeddings_generator import EmbeddingsGenerator
from vector_store import VectorStore
from rag_retriever import RAGretriever
from llm import LLM

def main():

    load_dotenv()

    # 1. Load PDF documents
    all_documents = PDFs_to_Documents("../Data")

    # 2. Split documents into chunks
    all_chunks = Documents_to_Chunks(all_documents)

    # 3. Generate embeddings
    embeddings_generator = EmbeddingsGenerator("all-MiniLM-L6-v2")
    text = [chunk.page_content for chunk in all_chunks]
    embeddings = (embeddings_generator.generate_embedding(text))

    # 4. Store embeddings in ChromaDB
    vector_store = VectorStore("pdf_RAG_embeddings","ModularVectorStore")
    vector_store.add_documents_embeddings(all_chunks,embeddings)

    # 5. Create retriever
    rag_retriever = RAGretriever(embeddings_generator,vector_store)

    # 6. Query
    query = "Give summary of Nupur Maheshwari of AI ML"
    results = rag_retriever.retrieve_results(query,3,0.0)

    # 7. Create context
    context = ("\n\n".join([result["page_content"] for result in results]) if results else "")

    # 8. Groq LLM
    llm = LLM("openai/gpt-oss-20b",os.getenv("GROQ_API_KEY"))

    # 9. Generate final answer
    answer = llm.generate_response(context,query)

    print("\n")
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)

if __name__ == "__main__":
    main()