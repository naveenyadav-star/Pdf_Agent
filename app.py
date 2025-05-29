from config.config import load_environment, get_env_variable
import streamlit as st
from retriever.retriever_query import retriver_query
from openai import AuthenticationError, RateLimitError, OpenAIError,APIStatusError
import requests
from utils.lang_smith_tracer import setup_langsmith_tracer

# Load environment variables
load_environment()
def ask_with_fallback(qa_chain, query,callbacks=None):
    """
    Invokes the QA chain and catches common API errors.
    """
    try:
        return qa_chain.invoke(query,config={"callbacks": callbacks})
    
    except AuthenticationError:
        st.error("🔒 Invalid or missing API key. Please check your key in `.env` or `secrets.toml`.")
    
    except RateLimitError:
        st.warning("⚠️ You've hit the rate limit or quota. Please wait or switch providers.")
    
    except APIStatusError as e:
        st.error(f"❗ Insufficient Balance: ")
    
    except OpenAIError as e:
        st.error(f"💥 OpenAI Error: {str(e)}")
    
    except requests.exceptions.HTTPError as e:
        # Groq or other LLMs may throw HTTPError
        status = e.response.status_code
        if status == 403:
            st.error("🚫 Access forbidden. Please check API permissions.")
        elif status == 401:
            st.error("🔐 Unauthorized. Check your API credentials.")
        else:
            st.error(f"🌐 HTTP Error {status}: {e.response.text}")
    
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {str(e)}")
    
    return None

pdf_path = get_env_variable("PDF_PATH", default="data/WEF_Future_of_Jobs_Report_2025.pdf") 

st.set_page_config(page_title="📄 RAG PDF QA", layout="wide")
st.title("📄 Ask Questions Related- AI and the Future of Work Jobs Report 2025 -PDF File")


if "qa_chain" not in st.session_state:
    with st.spinner("🔄 Loading Vector DB & Retrieval Chain..."):
        st.session_state["qa_chain"] = retriver_query(query=None)
qa_chain = st.session_state["qa_chain"]


# ask_section = st.empty()
query = st.text_input("❓ Your question:")
callbacks = setup_langsmith_tracer()

if query:
    with st.spinner("🤖 Thinking..."):
        answer = ask_with_fallback(st.session_state.qa_chain, query,callbacks=callbacks)
        # answer = st.session_state.qa_chain.invoke(query)
        # print(answer)
        if answer is not None:
            st.markdown(f"**🧠 Answer:** {answer['result']}")