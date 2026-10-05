from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def PDFs_to_Documents(path_directory):
    directory = Path(path_directory)
    PDF_files = list(directory.glob("**/*.pdf"))
    print(f"Found {len(PDF_files)} PDF files in the directory.")
    print()
    all_documents = []
    for PDF_file in PDF_files:
        print(f"Processing: {PDF_file.name}")
        try:
            loader = PyMuPDFLoader(str(PDF_file))
            documents = loader.load()
            for document in documents:
                document.metadata["source_file"] = PDF_file.name
                document.metadata["file_type"] = "pdf"
                document.metadata["author"] = "Nupur Maheshwari"
                document.metadata["date_created"] = "2026-10-20"
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} documents from {PDF_file.name}")
            print()
        except Exception as e:
            print("Error occurred while processing {PDF_file.name}: {e}")
            print()
    print(f"Total documents loaded: {len(all_documents)}")
    print()
    return all_documents

def Documents_to_Chunks(documents,chunk_size=600,chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    all_chunks = splitter.split_documents(documents)
    print(f"{len(documents)} documents split into {len(all_chunks)} chunks.")
    print()
    return all_chunks