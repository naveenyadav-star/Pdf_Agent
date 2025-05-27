
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import os

from config.config import load_environment, get_env_variable
# Load environment variables    
load_environment()
OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
CHROMA_DB_DIR = get_env_variable("CHROMA_DB_DIR")

def get_or_create_chroma_db(splits, embeddings, force_rebuild=False):

    
    
    
    if os.path.exists(CHROMA_DB_DIR) and not force_rebuild:
        print("📂 Chroma DB already exists. Loading from disk...")
        vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        # vectordb = vector_store
    else:
        print("💾 Creating and storing embeddings in Chroma DB...")
        vector_store = Chroma(
            collection_name="myAIDoc",
            # documents=splits,
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",  
        )
        vectordb = vector_store.add_documents(splits)
        vectordb.persist()
        print("✅ Chroma DB created and stored successfully.")

    return vectordb

def create_embedding_store(documents: str, chunk_size: int = 1000, chunk_overlap: int = 200,force_embed=False):
    """
    Create an embedding store from a PDF file.

    Args:

        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        RetrievalQA: A retrieval-based question-answering chain.
    """


    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY
    )



    vectordb=get_or_create_chroma_db(splits, embeddings, force_rebuild=False)

    # Create a retriever from the texts
    # retriever = embeddings.as_retriever()

    # Create a RetrievalQA chain
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    # )

    return vectordb