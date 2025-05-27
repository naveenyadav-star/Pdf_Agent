from config.config import load_environment, get_env_variable
from retriever.retriever_query import retriver_query
from openai import AuthenticationError, RateLimitError, OpenAIError,APIStatusError

# Load environment variables
load_environment()

pdf_path = get_env_variable("PDF_PATH", default="data/WEF_Future_of_Jobs_Report_2025.pdf") 

def main():

    qa_chain=retriver_query(pdf_path, query=None)

    print("🧠 Ask your questions (type 'exit' or 'quit' to stop):\n")

    while True:
        query = input("📝 Your question: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting. Goodbye!")
            break
        if not query:
            continue

        # response = qa_chain.run(query)
        try:
            print("🤖 Thinking...")
            response = qa_chain.invoke(query)
            print(f"🤖 Answer: {response['result']}\n")
        except (AuthenticationError, RateLimitError, OpenAIError) as e:
            print(f"❗ Error: {e['errror']}")
            continue
        




if __name__ == "__main__":
    main()
