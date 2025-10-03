"""
Frontend Module for PDF Chat Application
Handles Streamlit UI and user interactions
"""

import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from backend import (
    # Database operations
    init_metadata_db,
    save_thread_metadata,
    get_thread_metadata,
    get_all_threads,
    delete_thread_metadata,
    
    # Thread management
    create_new_thread,
    load_conversation,
    delete_thread_faiss_index,
    get_thread_faiss_path,
    
    # PDF processing
    process_pdfs,
    
    # Question answering
    generate_answer_stream,
    search_and_answer,
    save_message_to_checkpointer
)


# ==================== Session State Management ====================
def init_session_state():
    """Initialize all session state variables"""
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
            load_thread_messages(latest_thread[0])
        else:
            # Create a new thread
            reset_chat()


def reset_chat():
    """Reset chat by creating a new thread"""
    thread_id, thread_data = create_new_thread()
    
    st.session_state['current_thread_id'] = thread_id
    st.session_state['message_history'] = []
    
    # Add to session state threads
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = {}
    
    st.session_state['chat_threads'][thread_id] = thread_data


def load_thread_messages(thread_id: str):
    """Load messages from a specific thread"""
    messages = load_conversation(thread_id)
    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = 'user'
        else:
            role = 'assistant'
        temp_messages.append({'role': role, 'content': msg.content})
    st.session_state['message_history'] = temp_messages


# ==================== UI Components ====================
def render_sidebar():
    """Render the sidebar with all controls"""
    with st.sidebar:
        st.title("📁 PDF Chat Menu")
        
        # New Chat Button
        if st.button("➕ New Chat", use_container_width=True):
            reset_chat()
            st.rerun()
        
        st.divider()
        
        # PDF Upload Section
        render_pdf_upload_section()
        
        st.divider()
        
        # Chat History Section
        render_chat_history()
        
        st.divider()
        
        # Instructions
        render_instructions()


def render_pdf_upload_section():
    """Render PDF upload and processing section"""
    current_thread_id = st.session_state['current_thread_id']
    current_thread = st.session_state['chat_threads'].get(current_thread_id, {})
    
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
            handle_pdf_processing(pdf_docs, current_thread_id)
        else:
            st.warning("⚠️ Please upload at least one PDF file.")


def handle_pdf_processing(pdf_docs, thread_id: str):
    """Handle PDF upload and processing"""
    try:
        with st.spinner("Processing PDFs..."):
            # Show progress for each PDF
            for idx, pdf in enumerate(pdf_docs, 1):
                st.info(f"Processing PDF {idx}/{len(pdf_docs)}: {pdf.name}")
            
            # Process PDFs using backend
            pdf_names, chunk_count = process_pdfs(pdf_docs, thread_id)
            
            st.success(f"✅ Text extraction complete!")
            st.info(f"Created {chunk_count} text chunks")
            st.success(f"✅ FAISS index created successfully with {chunk_count} chunks!")
            
            # Update thread metadata
            metadata = get_thread_metadata(thread_id)
            if metadata:
                save_thread_metadata(
                    thread_id,
                    metadata['name'],
                    metadata['created_at'],
                    pdf_names,
                    True
                )
                
                # Update session state
                st.session_state['chat_threads'][thread_id]['pdf_files'] = pdf_names
                st.session_state['chat_threads'][thread_id]['has_faiss_index'] = True
            
            st.success("✅ Processing completed! You can now ask questions.")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error during processing: {e}")


def render_chat_history():
    """Render chat history with thread selection"""
    st.subheader("💬 My Conversations")
    
    if st.session_state['chat_threads']:
        # Sort threads by creation time (newest first)
        sorted_threads = sorted(
            st.session_state['chat_threads'].items(),
            key=lambda x: x[1]['created_at'],
            reverse=True
        )
        
        for thread_id, thread_data in sorted_threads:
            render_thread_item(thread_id, thread_data)
    else:
        st.info("No conversations yet. Start chatting!")


def render_thread_item(thread_id: str, thread_data: dict):
    """Render a single thread item in the sidebar"""
    is_current = thread_id == st.session_state['current_thread_id']
    
    # Show PDF count indicator
    pdf_count = len(thread_data.get('pdf_files', []))
    pdf_indicator = f" ({pdf_count} PDF{'s' if pdf_count != 1 else ''})" if pdf_count > 0 else " (No PDFs)"
    button_label = f"{'🔵 ' if is_current else ''}{thread_data['name']}{pdf_indicator}"
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        if st.button(button_label, key=f"thread_{thread_id}", use_container_width=True):
            st.session_state['current_thread_id'] = thread_id
            load_thread_messages(thread_id)
            st.rerun()
    
    with col2:
        if st.button("🗑️", key=f"del_{thread_id}"):
            handle_thread_deletion(thread_id)


def handle_thread_deletion(thread_id: str):
    """Handle thread deletion"""
    # Delete the thread's FAISS index
    delete_thread_faiss_index(thread_id)
    
    # Delete thread metadata from database
    delete_thread_metadata(thread_id)
    
    # Delete from session state
    del st.session_state['chat_threads'][thread_id]
    
    # If deleted thread was current, reset to new chat
    if thread_id == st.session_state['current_thread_id']:
        reset_chat()
    
    st.rerun()


def render_instructions():
    """Render instructions expander"""
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


def render_main_chat():
    """Render the main chat interface"""
    st.header("📄 Chat with PDF using Gemini 💁")
    
    current_thread_id = st.session_state['current_thread_id']
    current_thread = st.session_state['chat_threads'].get(current_thread_id, {})
    
    # Display FAISS index status
    faiss_path = get_thread_faiss_path(current_thread_id)
    if os.path.exists(faiss_path) and current_thread.get('has_faiss_index'):
        pdf_list = ", ".join(current_thread.get('pdf_files', []))
        st.success(f"✅ PDFs loaded for this chat: {pdf_list}")
    else:
        st.warning("⚠️ No PDFs uploaded for this chat. Please upload and process PDF files first.")
    
    # Display current thread name
    st.caption(f"Current conversation: {current_thread.get('name', 'Unknown')}")
    
    # Display message history
    display_message_history()
    
    # Chat input
    handle_chat_input(current_thread_id)


def display_message_history():
    """Display all messages in the chat history"""
    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])


def handle_chat_input(thread_id: str):
    """Handle user chat input"""
    if user_question := st.chat_input("Ask a question about your PDFs..."):
        # Add user message to history
        st.session_state['message_history'].append({
            'role': 'user', 
            'content': user_question
        })
        
        # Save to LangGraph checkpointer
        save_message_to_checkpointer(thread_id, user_question, is_user=True)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate and display assistant response
        generate_response(user_question, thread_id)


def generate_response(user_question: str, thread_id: str):
    """Generate and display AI response"""
    faiss_path = get_thread_faiss_path(thread_id)
    
    if not os.path.exists(faiss_path):
        st.error("❌ No PDFs uploaded for this chat. Please upload and process PDF files first.")
        return

    try:
        # Search for relevant context
        context, docs = search_and_answer(user_question, thread_id)
        
        if not context:
            st.warning("No relevant content found for your question.")
            return

        # Stream the answer
        with st.chat_message("assistant"):
            answer = st.write_stream(generate_answer_stream(context, user_question))
        
        # Save to message history
        st.session_state['message_history'].append({
            'role': 'assistant', 
            'content': answer
        })
        
        # Save to LangGraph checkpointer
        save_message_to_checkpointer(thread_id, answer, is_user=False)

        # Display source context
        display_source_context(docs)

    except Exception as e:
        st.error(f"Error processing your question: {e}")


def display_source_context(docs):
    """Display source documents used for answering"""
    with st.expander("📖 Source Context"):
        for i, doc in enumerate(docs):
            st.write(f"**Chunk {i+1}:**")
            content = doc.page_content
            display_content = content[:500] + "..." if len(content) > 500 else content
            st.write(display_content)
            st.write("---")


# ==================== Main Application ====================
def main():
    """Main application entry point"""
    # Configure page
    st.set_page_config(
        page_title="Chat with PDF - Multi-Thread",
        page_icon="📄",
        layout="wide"
    )

    # Initialize metadata database
    init_metadata_db()

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
        return

    # Initialize session state
    init_session_state()

    # Render UI
    render_sidebar()
    render_main_chat()


if __name__ == "__main__":
    main()