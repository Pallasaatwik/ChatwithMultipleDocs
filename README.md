# 💬 Chat with Multiple Documents

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatwithmultipledocs.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Interact with multiple documents in a conversational way using LLMs, FAISS for retrieval, and LangChain to manage the workflow.**

This Streamlit application allows you to upload PDFs, embed them using Google Gemini API (or OpenAI), and chat with them directly using advanced natural language processing techniques.

![App Screenshot](Screenshot%202025-07-27%20163428.png)

## 🚀 Features

- **📄 Multi-Document Support**: Upload and process multiple PDF documents simultaneously
- **🤖 AI-Powered Chat**: Conversational interface powered by Google Gemini or OpenAI
- **🔍 Semantic Search**: FAISS vector store for efficient document retrieval
- **💡 Context-Aware Responses**: LangChain framework ensures coherent, contextual answers
- **🎨 User-Friendly Interface**: Clean, intuitive Streamlit frontend
- **⚡ Fast Processing**: Optimized embedding and retrieval pipeline

## 🎯 Live Demo

🔗 **Try it now:** [Chat with Multiple Docs](https://chatwithmultipledocs.streamlit.app/)

## 📋 Table of Contents

- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Local Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Pallasaatwik/ChatwithMultipleDocs.git
   cd ChatwithMultipleDocs
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   ```

3. **Activate Virtual Environment**
   ```bash
   # On Linux/macOS:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

1. **Create Environment File**
   ```bash
   touch .env
   ```

2. **Add API Keys**
   ```env
   # Required: Google Gemini API Key
   GOOGLE_API_KEY=your_google_gemini_api_key_here
   
   # Optional: OpenAI API Key (if using OpenAI models)
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Get API Keys**
   - **Google Gemini**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **OpenAI** (optional): Visit [OpenAI API Keys](https://platform.openai.com/api-keys)

## 🚀 Usage

1. **Start the Application**
   ```bash
   streamlit run app.py
   ```

2. **Open in Browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL manually

3. **Using the App**
   - Upload one or more PDF documents using the file uploader
   - Wait for the documents to be processed and embedded
   - Start asking questions about your documents in the chat interface
   - Get contextual answers based on the content of your uploaded files

## 🧠 How It Works

```mermaid
graph LR
    A[Upload PDFs] --> B[Text Extraction]
    B --> C[Text Chunking]
    C --> D[Generate Embeddings]
    D --> E[Store in FAISS]
    E --> F[User Query]
    F --> G[Retrieve Relevant Chunks]
    G --> H[Generate Response]
    H --> I[Display Answer]
```

### Step-by-Step Process

1. **📄 Document Upload**: Users upload PDF documents through the Streamlit interface
2. **✂️ Text Splitting**: Documents are split into manageable chunks using LangChain's text splitters
3. **🔢 Embedding Generation**: Text chunks are converted to vector embeddings using Google Gemini API
4. **🗃️ Vector Storage**: Embeddings are stored in a FAISS index for efficient similarity search
5. **💬 Query Processing**: User questions are embedded and matched against stored document vectors
6. **🎯 Context Retrieval**: Most relevant document chunks are retrieved based on semantic similarity
7. **🤖 Response Generation**: LangChain orchestrates the LLM to generate contextual responses

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | [Streamlit](https://streamlit.io/) | Web interface and user interaction |
| **LLM Framework** | [LangChain](https://langchain.com/) | Document processing and chat orchestration |
| **Embeddings** | [Google Gemini API](https://ai.google.dev/) | Text-to-vector conversion |
| **Vector Store** | [FAISS](https://faiss.ai/) | Efficient similarity search |
| **PDF Processing** | [PyPDF2](https://pypdf2.readthedocs.io/) | PDF text extraction |
| **Environment** | Python 3.8+ | Runtime environment |

## 📁 Project Structure

```
ChatwithMultipleDocs/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
├── Screenshot 2025-07-27 163428.png  # App screenshot
│
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── document_processor.py  # PDF processing logic
│   ├── embeddings.py         # Embedding generation
│   └── chat_handler.py       # Chat functionality
│
├── tests/                # Unit tests
│   ├── __init__.py
│   └── test_app.py
│
└── docs/                 # Additional documentation
    ├── deployment.md
    └── api_guide.md
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**
   ```bash
   git fork https://github.com/Pallasaatwik/ChatwithMultipleDocs.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Add your improvements
   - Write tests for new features
   - Update documentation as needed

4. **Commit Changes**
   ```bash
   git commit -m "Add: your feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include unit tests for new features
- Update README.md if needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Known Issues

- Large PDF files (>100MB) may take longer to process
- Some scanned PDFs might not extract text properly
- Rate limiting may apply for API calls

## 🔮 Roadmap

- [ ] Support for more document formats (DOCX, TXT, etc.)
- [ ] Multi-language support
- [ ] Document summarization features
- [ ] Export chat history
- [ ] Advanced filtering options
- [ ] Docker containerization

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Pallasaatwik/ChatwithMultipleDocs/issues) page
2. Create a new issue with detailed information
3. Contact the author directly

## 👤 Author

**Palla Saatwik Reddy**

- 🌐 GitHub: [@Pallasaatwik](https://github.com/Pallasaatwik)
- 📧 Email: saatwikreddy37@gmail.com
- 💼 LinkedIn: [Palla Saatwik Reddy](https://www.linkedin.com/in/saatwikreddy37/)

---

### ⭐ Star this repository if you found it helpful!
