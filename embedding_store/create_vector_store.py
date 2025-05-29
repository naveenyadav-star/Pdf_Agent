
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import os
import shutil
from config.config import load_environment, get_env_variable
# Load environment variables    
load_environment()
OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY") 
CHROMA_DB_DIR = get_env_variable("CHROMA_DB_DIR", default="./chroma_langchain_db")  # Default directory if not set
DELETE_VECTOR_STORE = get_env_variable("DELETE_VECTOR_STORE", default="False").lower() == "true"
pdf_path = get_env_variable("PDF_PATH", default="data/The_opportunities_and_risks_of_employment.pdf")
JOBS_PDF_COLLECTION_NAME = get_env_variable("JOBS_PDF_COLLECTION_NAME", default="myAIDoc")
from utils.read_pdf import read_pdf_file


def load_existing_vector_store(embeddings):
    print("📦 Loading existing vector store...")
    vector_store=Chroma(
        collection_name=JOBS_PDF_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    print("✅ Done existing vector store...")

    return vector_store

def get_or_create_chroma_db( embeddings,DELETE_VECTOR_STORE=False,chunk_size: int = 2000, chunk_overlap: int = 500,JOBS_PDF_COLLECTION_NAME: str = "myAIDoc"):
    
    
    documents=read_pdf_file(pdf_path) 

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)
   
    print("💾 Creating and storing embeddings in Chroma DB...")

    vector_store = Chroma.from_documents(
            collection_name=JOBS_PDF_COLLECTION_NAME,
            documents=splits,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
   
    print("💾 Chroma DB created and stored successfully.✅")

    return vector_store

def create_vector_store(chunk_size: int = 2000, chunk_overlap: int = 500, force_embed=False):
    """
    Create an embedding store from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        Chroma: A vector store containing the embeddings.
    """
    if DELETE_VECTOR_STORE and os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        print("🗑️ Existing vector store deleted.")
    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY
    )

    # Either create new or load existing vector store
    if os.path.exists(CHROMA_DB_DIR):
        print("📦 Loading existing vector store...")
        vector_store=load_existing_vector_store(embeddings)
    else:
        print("📦 Creating vector store...")
        vector_store= get_or_create_chroma_db(embeddings,DELETE_VECTOR_STORE=False,chunk_size = 2000, chunk_overlap = 500,JOBS_PDF_COLLECTION_NAME=JOBS_PDF_COLLECTION_NAME )
        

    return vector_store

