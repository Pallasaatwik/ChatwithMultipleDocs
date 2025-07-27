# 💬 Chat with Multiple Documents using LangChain + FAISS + Streamlit

Interact with multiple documents in a conversational way using LLMs, FAISS for retrieval, and LangChain to manage the workflow. This Streamlit app allows you to upload PDFs, embed them using Gemini API (or OpenAI), and chat with them directly!

![App Screenshot](Screenshot%202025-07-27%20163428.png)

🔗 **Live App:** [Chat with Multiple Docs](https://chatwithmultipledocs.streamlit.app/)

---

⚙️ Setup Instructions (Local)
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/Pallasaatwik/ChatwithMultipleDocs.git
cd ChatwithMultipleDocs
2. Create and Activate a Virtual Environment
bash
Copy
Edit
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Set Environment Variables
Create a .env file in the root directory and add your API key:

env
Copy
Edit
GOOGLE_API_KEY=your_google_gemini_api_key
5. Run the App
bash
Copy
Edit
streamlit run app.py

🧠 How it Works
Upload PDFs: The app reads and splits your uploaded PDF documents.

Embedding: Text chunks are embedded using Google Gemini or OpenAI.

FAISS Indexing: Embeddings are stored in a FAISS index for semantic search.

Chat: LangChain enables context-aware chat based on retrieved documents.

🛠️ Tech Stack
Frontend: Streamlit

LLM Framework: LangChain

Embedding Models: Google Gemini API / OpenAI

Vector Store: FAISS

PDF Parsing: PyPDF2

Environment Management: Python venv

👤 Author
Palla Saatwik Reddy
GitHub: @Pallasaatwik

