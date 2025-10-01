
import streamlit as st
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
from typing import TypedDict, Annotated
import gc
import uuid
import sqlite3
from datetime import datetime

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- LangGraph Setup ---
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chat_node(state: ChatState):
    # This is a placeholder - actual chat happens through user_input function
    return {"messages": state["messages"]}

# Initialize SQLite connection and checkpointer
conn = sqlite3.connect(database='pdf_chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

# --- Thread Metadata Database Functions ---
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

def save_thread_metadata(thread_id, name, created_at, pdf_files, has_faiss_index):
    """Save or update thread metadata"""
    cursor = conn.cursor()
    pdf_files_str = ','.join(pdf_files) if pdf_files else ''
    cursor.execute('''
        INSERT OR REPLACE INTO thread_metadata 
        (thread_id, name, created_at, pdf_files, has_faiss_index)
        VALUES (?, ?, ?, ?, ?)
    ''', (thread_id, name, created_at, pdf_files_str, 1 if has_faiss_index else 0))
    conn.commit()

def get_thread_metadata(thread_id):
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

def get_all_threads():
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

def delete_thread_metadata(thread_id):
    """Delete thread metadata from database"""
    cursor = conn.cursor()
    cursor.execute('DELETE FROM thread_metadata WHERE thread_id = ?', (thread_id,))
    conn.commit()

def retrieve_all_thread_ids():
    """Get list of all thread IDs"""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

# --- Utility Functions for Chat Threading ---
def generate_thread_id():
    return str(uuid.uuid4())

def get_thread_faiss_path(thread_id):
    """Get the FAISS index path for a specific thread"""
    return f"faiss_index_{thread_id}"

def reset_chat():
    thread_id = generate_thread_id()
    thread_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    created_at = datetime.now().isoformat()
    
    st.session_state['current_thread_id'] = thread_id
    st.session_state['message_history'] = []
    
    # Save metadata to database
    save_thread_metadata(thread_id, thread_name, created_at, [], False)
    
    # Add to session state threads
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = {}
    
    st.session_state['chat_threads'][thread_id] = {
        'name': thread_name,
        'created_at': created_at,
        'pdf_files': [],
        'has_faiss_index': False
    }

def load_conversation(thread_id):
    """Load conversation from LangGraph checkpointer"""
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        return state.values.get('messages', [])
    except:
        return []

def delete_thread_faiss_index(thread_id):
    """Delete FAISS index for a specific thread"""
    faiss_path = get_thread_faiss_path(thread_id)
    if os.path.exists(faiss_path):
        try:
            def remove_readonly(func, path, _):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(faiss_path, onerror=remove_readonly)
        except Exception as e:
            st.error(f"Error deleting FAISS index: {e}")

# --- Extract text from PDF ---
def get_pdf_text(pdf_docs):
    text = ""
    pdf_names = []
    for pdf_idx, pdf in enumerate(pdf_docs):
        st.info(f"Processing PDF {pdf_idx + 1}/{len(pdf_docs)}: {pdf.name}")
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
            st.error(f"Failed to process PDF {pdf.name}: {e}")
        finally:
            try:
                os.unlink(temp_pdf_path)
            except Exception:
                pass

    st.success("✅ Text extraction complete!")
    return text, pdf_names

# --- Split into chunks ---
def get_text_chunks(text):
    if not text.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n--- Page", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    st.info(f"Created {len(chunks)} text chunks")
    return chunks

# --- Create FAISS index for specific thread ---
def get_vector_store(text_chunks, thread_id):
    def remove_readonly(func, path, _):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    gc.collect()

    faiss_path = get_thread_faiss_path(thread_id)
    
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
        st.success(f"✅ FAISS index created successfully with {len(text_chunks)} chunks!")
        
    except Exception as e:
        st.error(f"Failed to create FAISS index: {e}")
        raise

# --- Generate answer using Gemini with streaming ---
def generate_answer_stream(context, question):
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

# --- Handle user query ---
def user_input(user_question, thread_id):
    faiss_path = get_thread_faiss_path(thread_id)
    
    if not os.path.exists(faiss_path):
        st.error("❌ No PDFs uploaded for this chat. Please upload and process PDF files first.")
        return

    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        new_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=5)

        if not docs:
            st.warning("No relevant content found for your question.")
            return

        context = "\n\n".join([doc.page_content for doc in docs])

        # Stream the answer
        with st.chat_message("assistant"):
            answer = st.write_stream(generate_answer_stream(context, user_question))
        
        # Save to message history
        st.session_state['message_history'].append({'role': 'assistant', 'content': answer})
        
        # Save to LangGraph checkpointer
        CONFIG = {
            "configurable": {"thread_id": thread_id},
            "metadata": {"thread_id": thread_id},
            "run_name": "chat_turn",
        }
        
        # Update the chatbot state with the conversation
        chatbot.update_state(
            CONFIG,
            {"messages": [AIMessage(content=answer)]}
        )

        with st.expander("📖 Source Context"):
            for i, doc in enumerate(docs):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                st.write("---")

    except Exception as e:
        st.error(f"Error processing your question: {e}")

# --- Main app ---
def main():
    st.set_page_config(
        page_title="Chat with PDF - Multi-Thread",
        page_icon="📄",
        layout="wide"
    )

    # Initialize metadata database
    init_metadata_db()

    # Initialize session state
    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []
    
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = get_all_threads()
    
    if 'current_thread_id' not in st.session_state:
        if st.session_state['chat_threads']:
            # Load the most recent thread
            latest_thread = max(
                st.session_state['chat_threads'].items(),
                key=lambda x: x[1]['created_at']
            )
            st.session_state['current_thread_id'] = latest_thread[0]
            
            # Load conversation from checkpointer
            messages = load_conversation(latest_thread[0])
            temp_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = 'user'
                else:
                    role = 'assistant'
                temp_messages.append({'role': role, 'content': msg.content})
            st.session_state['message_history'] = temp_messages
        else:
            # Create a new thread
            reset_chat()

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
        return

    current_thread_id = st.session_state['current_thread_id']
    current_thread = st.session_state['chat_threads'].get(current_thread_id, {})

    # Sidebar
    with st.sidebar:
        st.title("📁 PDF Chat Menu")
        
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True):
            reset_chat()
            st.rerun()
        
        st.divider()
        
        # PDF Upload Section - Thread-specific
        st.subheader("📤 Upload PDFs (This Chat)")
        
        # Show uploaded PDFs for current thread
        if current_thread.get('pdf_files'):
            st.caption("Uploaded PDFs:")
            for pdf_name in current_thread['pdf_files']:
                st.text(f"📄 {pdf_name}")
        else:
            st.caption("No PDFs uploaded yet")
        
        pdf_docs = st.file_uploader(
            "Choose PDF Files", 
            accept_multiple_files=True,
            type=['pdf'],
            key=f"uploader_{current_thread_id}"
        )
        
        if st.button("🚀 Submit & Process", use_container_width=True):
            if pdf_docs:
                try:
                    with st.spinner("Processing PDFs..."):
                        raw_text, pdf_names = get_pdf_text(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("❌ No text found in the uploaded PDF files.")
                            return
                        
                        text_chunks = get_text_chunks(raw_text)
                        
                        if not text_chunks:
                            st.error("❌ No text chunks created.")
                            return
                        
                        # Create FAISS index for this specific thread
                        get_vector_store(text_chunks, current_thread_id)
                        
                        # Update thread metadata
                        metadata = get_thread_metadata(current_thread_id)
                        if metadata:
                            save_thread_metadata(
                                current_thread_id,
                                metadata['name'],
                                metadata['created_at'],
                                pdf_names,
                                True
                            )
                            
                            # Update session state
                            st.session_state['chat_threads'][current_thread_id]['pdf_files'] = pdf_names
                            st.session_state['chat_threads'][current_thread_id]['has_faiss_index'] = True
                        
                        st.success("✅ Processing completed! You can now ask questions.")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error during processing: {e}")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")
        
        st.divider()
        
        # Chat History Section
        st.subheader("💬 My Conversations")
        
        if st.session_state['chat_threads']:
            # Sort threads by creation time (newest first)
            sorted_threads = sorted(
                st.session_state['chat_threads'].items(),
                key=lambda x: x[1]['created_at'],
                reverse=True
            )
            
            for thread_id, thread_data in sorted_threads:
                # Highlight current thread
                is_current = thread_id == st.session_state['current_thread_id']
                
                # Show PDF count indicator
                pdf_count = len(thread_data.get('pdf_files', []))
                pdf_indicator = f" ({pdf_count} PDF{'s' if pdf_count != 1 else ''})" if pdf_count > 0 else " (No PDFs)"
                button_label = f"{'🔵 ' if is_current else ''}{thread_data['name']}{pdf_indicator}"
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(button_label, key=f"thread_{thread_id}", use_container_width=True):
                        st.session_state['current_thread_id'] = thread_id
                        
                        # Load conversation from checkpointer
                        messages = load_conversation(thread_id)
                        temp_messages = []
                        for msg in messages:
                            if isinstance(msg, HumanMessage):
                                role = 'user'
                            else:
                                role = 'assistant'
                            temp_messages.append({'role': role, 'content': msg.content})
                        st.session_state['message_history'] = temp_messages
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"del_{thread_id}"):
                        # Delete the thread's FAISS index
                        delete_thread_faiss_index(thread_id)
                        
                        # Delete thread metadata from database
                        delete_thread_metadata(thread_id)
                        
                        # Delete from session state
                        del st.session_state['chat_threads'][thread_id]
                        
                        if thread_id == st.session_state['current_thread_id']:
                            reset_chat()
                        st.rerun()
        else:
            st.info("No conversations yet. Start chatting!")
        
        st.divider()
        
        # Instructions
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Upload PDFs**: Choose PDF files for THIS chat only
            2. **Process**: Click 'Submit & Process' to extract text
            3. **Ask Questions**: Type your question in the chat
            4. **New Chat**: Start a fresh conversation with different PDFs
            5. **Resume**: Click on any conversation to continue
            
            **Note**: Each chat has its own PDFs. When you switch chats,
            you'll only have access to the PDFs uploaded in that specific chat.
            All conversations are stored in SQLite database.
            """)

    # Main Chat Interface
    st.header("📄 Chat with PDF using Gemini 💁")
    
    # Display FAISS index status for current thread
    faiss_path = get_thread_faiss_path(current_thread_id)
    if os.path.exists(faiss_path) and current_thread.get('has_faiss_index'):
        pdf_list = ", ".join(current_thread.get('pdf_files', []))
        st.success(f"✅ PDFs loaded for this chat: {pdf_list}")
    else:
        st.warning("⚠️ No PDFs uploaded for this chat. Please upload and process PDF files first.")
    
    # Display current thread name
    st.caption(f"Current conversation: {current_thread.get('name', 'Unknown')}")
    
    # Display message history
    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    if user_question := st.chat_input("Ask a question about your PDFs..."):
        # Add user message to history
        st.session_state['message_history'].append({'role': 'user', 'content': user_question})
        
        # Save to LangGraph checkpointer
        CONFIG = {
            "configurable": {"thread_id": current_thread_id},
            "metadata": {"thread_id": current_thread_id},
            "run_name": "chat_turn",
        }
        
        chatbot.update_state(
            CONFIG,
            {"messages": [HumanMessage(content=user_question)]}
        )
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate and display assistant response
        user_input(user_question, current_thread_id)

if __name__ == "__main__":
    main()