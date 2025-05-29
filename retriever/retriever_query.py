from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from utils.read_pdf import read_pdf_file
from embedding_store.embed_store import create_embedding_store  
from embedding_store.create_vector_store import create_vector_store
from langchain_groq.chat_models import ChatGroq
from config.config import load_environment, get_env_variable    
from langchain.prompts import PromptTemplate

# Load environment variables    

load_environment()

llm_flag = get_env_variable("LLM_FLAG", default="openai")  # "openai" or "groq"
DEEPSEEK_API_KEY = get_env_variable("DEEPSEEK_API_KEY" )  # Optional, if using DeepSeek

SUMMARY_RAG_PROMPT = PromptTemplate.from_template("""
Use ONLY the following extracted content to answer the question. 
If the answer is not found in the pdf content , but instead of don't know, respond briefly and politely with this.

Content:
{context}

Question: {question}

Answer in 50 words or less.
""")

def llm():

    if llm_flag == "openai":
        llm= ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    elif llm_flag == "groq":
        llm = ChatGroq(
        model_name="llama3-8b-8192",  # or "llama3-70b-8192"
        temperature=0,
        )
    else:
        llm = ChatOpenAI(
        model_name="deepseek-chat",  # or your actual DeepSeek model
        base_url="https://api.deepseek.com/v1",  # Replace with actual if self-hosted or official
        api_key=DEEPSEEK_API_KEY,
        temperature=0
        )
        

    return llm
    

def retriver_query(pdf_path: str, query: str):

    
    
    vectorstore = create_vector_store(pdf_path, force_embed=False)
    # Create a retriever from the vector store
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    # Build RAG chain (Prompt + Retriever)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm(),
        # template="You are a helpful assistant. Answer the question based on the provided context.",
        chain_type_kwargs={"prompt": SUMMARY_RAG_PROMPT},
        retriever=retriever,
        return_source_documents=True
    )


    return qa_chain
