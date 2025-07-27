# Chat with PDF using Gemini 🤖📄

A Streamlit web app that lets you chat with one or more PDF documents using Google Gemini AI. Upload your PDFs, ask questions, and get answers based on the document content—all in a sleek and user-friendly interface.

## 🚀 Live Demo

🔗 [Click here to use the app](https://pallasaatwik-chatwithmultipledocs.streamlit.app/)

## 📸 Screenshot

![App Screenshot](./a6cc7058-4c6b-4155-b519-df1dda0ff52c.png)

## ✨ Features

- 🗂️ Upload single or multiple PDF documents
- 🤖 Ask questions related to the documents
- 🔍 Intelligent and contextual answers powered by Google Gemini
- ⚡ Fast and interactive with real-time response
- 📂 Chunking and vector storage using FAISS
- 📎 File preview and content display

## 🧠 Tech Stack

- **Frontend & UI**: Streamlit
- **Embeddings**: `GoogleGenerativeAIEmbeddings` from `langchain_google_genai`
- **LLM**: Gemini Pro via `ChatGoogleGenerativeAI`
- **Vector Store**: FAISS
- **PDF Reading**: PyPDF2

## 🛠️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Pallasaatwik/ChatwithMultipleDocs.git
   cd ChatwithMultipleDocs
2. python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
Install dependencies


This project is licensed under the MIT License.
