# 📄 RAG-Powered PDF QA App

This Streamlit app allows users to **ask questions from a specific PDF** (like a report or whitepaper). It uses **Retrieval-Augmented Generation (RAG)** with OpenAI/Groq/Deepseek models and ChromaDB for local vector storage.

---

## 🚀 Features

- 🔍 Ask natural language questions based on a pre-defined PDF
- 🧠 Answers are extracted only from the document (not hallucinated)
- ⚡ Fast response using embedded vector store (ChromaDB)
- 📎 Support for models via OpenAI, Groq, DeepSeek
- 📊 Usage logging (optional Google Sheet support)
- 🧪 Smart prompts and fallback handling (Hi/How are you responses)
- ✅ Easily extendable with LangChain agents and memory

---

## 📁 Project Structure

├── app.py # Streamlit app entry
├── config/
│ └── config.py # Environment variable loader
├── retriever/
│ └── retriever_query.py # Core RAG logic
├── utils/
│ └── helper.py # Optional utilities
├── data/
│ └── WEF_Future_of_Jobs_Report_2025.pdf
├── .env # API Keys (Not shared)
├── requirements.txt
└── README.md

To RUN------------- streamlit run app.py

🧠 Example Questions for WEF Report
What are the top 10 emerging skills?

Which jobs are declining?

What are the findings about AI impact?

What are recommendations for reskilling?

📌 Notes
Only one static PDF is supported in this version.

Answers are derived from vector-based search.

“Hi”, “Hello”, and non-document questions will be handled by the base model (fallback).

Large PDFs (~290 pages) are chunked efficiently.