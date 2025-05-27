from langchain.callbacks.tracers.langchain import LangChainTracer



def setup_langsmith_tracer():
    import os
    from langchain.callbacks.tracers import LangChainTracer
    from langchain_core.tracers import ConsoleCallbackHandler
    from config.config import load_environment, get_env_variable
    # Load environment variables    
    load_environment()
    LANGCHAIN_PROJECT = get_env_variable("LANGCHAIN_PROJECT")
    LANGCHAIN_API_KEY = get_env_variable("LANGCHAIN_API_KEY")

    tracer = LangChainTracer(project_name=LANGCHAIN_PROJECT)
    return [ConsoleCallbackHandler(), tracer]
