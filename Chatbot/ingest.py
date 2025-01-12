import os
from langchain_community.document_loaders import CSVLoader  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Correct import for text_splitter
from langchain_huggingface import HuggingFaceEmbeddings  # Corrected import for embeddings
from langchain_community.vectorstores import FAISS  # Updated import for FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
import pickle

def load_documents():
    file_paths = [
        'Bhagwad_Gita_Verses_English_Questions.csv',
        'Patanjali_Yoga_Sutras_Verses_English_Questions.csv'
    ]
    documents = []
    for file_path in file_paths:
        try:
            # Use 'utf-8' encoding to handle potential encoding issues
            loader = CSVLoader(file_path=file_path, encoding='utf-8')
            documents.extend(loader.load())
            print(f"Loaded documents from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    return documents

def split_texts(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    print(f"Total documents split into {len(texts)} chunks.")
    return texts

def create_vector_db():
    documents = load_documents()  # Load documents
    if not documents:
        print("No documents to process.")
        return
    texts = split_texts(documents)  # Split into text chunks

    # Load HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create FAISS vector database
    db = FAISS.from_documents(texts, embeddings)
    print("Vector database created successfully.")

    # Save vector database to disk
    with open("vector_db.pkl", "wb") as f:
        pickle.dump(db, f)
    print("Vector database saved to disk.")

if __name__ == "__main__":
    create_vector_db()
