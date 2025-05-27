# config.py
import os
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st

def load_environment(dotenv_path: str = ".env"):
    """
    Loads environment variables from the specified .env file.
    Raises an error if the file or required keys are missing.
    """
    env_path = Path(dotenv_path)
    
    # if not env_path.exists():
    #     raise FileNotFoundError(f"{dotenv_path} file not found.")

    # load_dotenv(dotenv_path)

    if not st.secrets:  # Only load .env if st.secrets is empty
        env_path = Path(dotenv_path)
        if env_path.exists():
            load_dotenv(dotenv_path)
        else:
            raise FileNotFoundError(f"{dotenv_path} file not found.")

    # Validate required environment variables
    required_keys = ["OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_keys)}")

def get_env_variable(key: str, default: str = None) -> str:
    """
    Safely gets an environment variable with an optional default fallback.
    """
    # value = os.getenv(key, default)
    if key in st.secrets:
        return st.secrets.get(key)
    value = os.getenv(key, default)
    if value is None:
        raise EnvironmentError(f"Environment variable '{key}' is not set and no default was provided.")
    return value
