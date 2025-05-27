from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from utils.read_pdf import read_pdf_file
from embedding_store.embed_store import create_embedding_store  
from langchain_groq.chat_models import ChatGroq
from config.config import load_environment, get_env_variable    
from langchain.prompts import PromptTemplate
# Load environment variables    

load_environment()

llm_flag = get_env_variable("LLM_FLAG", default="openai")  # "openai" or "groq"
DEEPSEEK_API_KEY = get_env_variable("DEEPSEEK_API_KEY" )  # Optional, if using DeepSeek

SUMMARY_RAG_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant. If the question is not about the uploaded PDF or documentation, but instead friendly talk, respond briefly and politely with minimum.

---
Context:
{context}

Question:
{question}

Answer:
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
        temperature=0.7
        )
        

    return llm
    

def retriver_query(pdf_path: str, query: str):

    documents=read_pdf_file(pdf_path) 

    vectorstore = create_embedding_store(documents, force_embed=False)
    # Create a retriever from the vector store
        

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    # Build RAG chain (Prompt + Retriever)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm(),
        # template="You are a helpful assistant. Answer the question based on the provided context.",
        chain_type_kwargs={"prompt": SUMMARY_RAG_PROMPT},
        retriever=retriever,
        return_source_documents=True
    )


    return qa_chain
