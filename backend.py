"""
Backend Module for PDF Chat Application
Handles PDF processing, vector store management, and database operations
"""

import fitz  # PyMuPDF
import os
import tempfile
import shutil
import stat
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Dict, Generator
import gc
import uuid
import sqlite3
from datetime import datetime

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ==================== LangGraph Setup ====================
class ChatState(TypedDict):
    """State definition for chat graph"""
    messages: Annotated[list, add_messages]


def chat_node(state: ChatState):
    """Chat node for LangGraph - placeholder for graph structure"""
    return {"messages": state["messages"]}


# Initialize SQLite connection and checkpointer
conn = sqlite3.connect(database='pdf_chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Build the graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)


# ==================== Database Operations ====================
def init_metadata_db():
    """Initialize metadata table for thread information"""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS thread_metadata (
            thread_id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT,
            pdf_files TEXT,
            has_faiss_index INTEGER
        )
    ''')
    conn.commit()


def save_thread_metadata(thread_id: str, name: str, created_at: str, 
                        pdf_files: List[str], has_faiss_index: bool):
    """Save or update thread metadata"""
    cursor = conn.cursor()
    pdf_files_str = ','.join(pdf_files) if pdf_files else ''
    cursor.execute('''
        INSERT OR REPLACE INTO thread_metadata 
        (thread_id, name, created_at, pdf_files, has_faiss_index)
        VALUES (?, ?, ?, ?, ?)
    ''', (thread_id, name, created_at, pdf_files_str, 1 if has_faiss_index else 0))
    conn.commit()


def get_thread_metadata(thread_id: str) -> Dict:
    """Retrieve thread metadata"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT name, created_at, pdf_files, has_faiss_index 
        FROM thread_metadata WHERE thread_id = ?
    ''', (thread_id,))
    result = cursor.fetchone()
    if result:
        return {
            'name': result[0],
            'created_at': result[1],
            'pdf_files': result[2].split(',') if result[2] else [],
            'has_faiss_index': bool(result[3])
        }
    return None


def get_all_threads() -> Dict:
    """Retrieve all thread IDs and metadata"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT thread_id, name, created_at, pdf_files, has_faiss_index 
        FROM thread_metadata 
        ORDER BY created_at DESC
    ''')
    threads = {}
    for row in cursor.fetchall():
        threads[row[0]] = {
            'name': row[1],
            'created_at': row[2],
            'pdf_files': row[3].split(',') if row[3] else [],
            'has_faiss_index': bool(row[4])
        }
    return threads


def delete_thread_metadata(thread_id: str):
    """Delete thread metadata from database"""
    cursor = conn.cursor()
    cursor.execute('DELETE FROM thread_metadata WHERE thread_id = ?', (thread_id,))
    conn.commit()


# ==================== Thread Management ====================
def generate_thread_id() -> str:
    """Generate a unique thread ID"""
    return str(uuid.uuid4())


def get_thread_faiss_path(thread_id: str) -> str:
    """Get the FAISS index path for a specific thread"""
    return f"faiss_index_{thread_id}"


def create_new_thread() -> tuple:
    """Create a new thread with metadata"""
    thread_id = generate_thread_id()
    thread_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    created_at = datetime.now().isoformat()
    
    # Save metadata to database
    save_thread_metadata(thread_id, thread_name, created_at, [], False)
    
    return thread_id, {
        'name': thread_name,
        'created_at': created_at,
        'pdf_files': [],
        'has_faiss_index': False
    }


def load_conversation(thread_id: str) -> List:
    """Load conversation from LangGraph checkpointer"""
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        return state.values.get('messages', [])
    except:
        return []


def delete_thread_faiss_index(thread_id: str):
    """Delete FAISS index for a specific thread"""
    faiss_path = get_thread_faiss_path(thread_id)
    if os.path.exists(faiss_path):
        try:
            def remove_readonly(func, path, _):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(faiss_path, onerror=remove_readonly)
        except Exception as e:
            raise Exception(f"Error deleting FAISS index: {e}")


# ==================== PDF Processing ====================
def extract_text_from_pdf(pdf_docs) -> tuple:
    """Extract text from uploaded PDF files"""
    text = ""
    pdf_names = []
    
    for pdf_idx, pdf in enumerate(pdf_docs):
        pdf_names.append(pdf.name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            temp_pdf_path = tmp_file.name

        try:
            doc = fitz.open(temp_pdf_path)
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_text = page.get_text()
                if page_text.strip():
                    text += f"\n--- Page {page_index+1} ---\n{page_text}\n"
            doc.close()
        except Exception as e:
            raise Exception(f"Failed to process PDF {pdf.name}: {e}")
        finally:
            try:
                os.unlink(temp_pdf_path)
            except Exception:
                pass

    return text, pdf_names


def split_text_into_chunks(text: str) -> List[str]:
    """Split text into chunks for processing"""
    if not text.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n--- Page", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_store(text_chunks: List[str], thread_id: str):
    """Create FAISS index for specific thread"""
    def remove_readonly(func, path, _):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    gc.collect()
    faiss_path = get_thread_faiss_path(thread_id)
    
    # Remove existing index if present
    if os.path.exists(faiss_path):
        try:
            shutil.rmtree(faiss_path, onerror=remove_readonly)
        except PermissionError:
            temp_name = f"{faiss_path}_old_{os.getpid()}"
            os.rename(faiss_path, temp_name)
            try:
                shutil.rmtree(temp_name, onerror=remove_readonly)
            except Exception:
                pass

    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        os.makedirs(faiss_path, exist_ok=True)
        vector_store.save_local(faiss_path)
        return len(text_chunks)
    except Exception as e:
        raise Exception(f"Failed to create FAISS index: {e}")


# ==================== Question Answering ====================
def generate_answer_stream(context: str, question: str) -> Generator:
    """Generate answer using Gemini with streaming"""
    try:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

        prompt = f"""
        You are an AI assistant that can answer questions about PDF documents.
        The context below is extracted text from the PDF.

        Answer the question as detailed as possible from the provided context.
        If the answer is not in the provided context, say: "Answer is not available in the context".
        Don't guess or make up information.

        Context:
        {context}

        Question: {question}

        Answer:
        """

        for chunk in model.stream(prompt):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        yield f"Error generating answer: {e}"


def search_and_answer(user_question: str, thread_id: str) -> tuple:
    """Search FAISS index and generate answer"""
    faiss_path = get_thread_faiss_path(thread_id)
    
    if not os.path.exists(faiss_path):
        raise Exception("No PDFs uploaded for this chat")

    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        new_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=5)

        if not docs:
            return None, []

        context = "\n\n".join([doc.page_content for doc in docs])
        return context, docs

    except Exception as e:
        raise Exception(f"Error processing your question: {e}")


def save_message_to_checkpointer(thread_id: str, message, is_user: bool):
    """Save message to LangGraph checkpointer"""
    CONFIG = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }
    
    if is_user:
        msg = HumanMessage(content=message)
    else:
        msg = AIMessage(content=message)
    
    chatbot.update_state(CONFIG, {"messages": [msg]})


def process_pdfs(pdf_docs, thread_id: str) -> tuple:
    """Complete PDF processing pipeline"""
    # Extract text
    raw_text, pdf_names = extract_text_from_pdf(pdf_docs)
    
    if not raw_text.strip():
        raise Exception("No text found in the uploaded PDF files")
    
    # Split into chunks
    text_chunks = split_text_into_chunks(raw_text)
    
    if not text_chunks:
        raise Exception("No text chunks created")
    
    # Create vector store
    chunk_count = create_vector_store(text_chunks, thread_id)
    
    return pdf_names, chunk_count