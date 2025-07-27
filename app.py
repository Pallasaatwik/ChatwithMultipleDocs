import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Create directory if it doesn't exist
        os.makedirs("faiss_index", exist_ok=True)
        
        vector_store.save_local("faiss_index")
        st.success(f"FAISS index created successfully with {len(text_chunks)} chunks!")
        return True
    except Exception as e:
        st.error(f"Error creating FAISS index: {str(e)}")
        return False

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        # Check if FAISS index exists
        if not os.path.exists("faiss_index"):
            st.error("No FAISS index found. Please upload and process PDF files first.")
            return
            
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)

        print(response)
        st.write("Reply: ", response["output_text"])
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    
    # Show current status
    if os.path.exists("faiss_index"):
        st.success("✅ FAISS index is ready!")
    else:
        st.warning("⚠️ Please upload and process PDF files first.")
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        # Extract text
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("No text found in the uploaded PDF files.")
                            return
                        
                        st.info(f"Extracted {len(raw_text)} characters from PDFs")
                        
                        # Create chunks
                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                            st.error("No text chunks created.")
                            return
                            
                        st.info(f"Created {len(text_chunks)} text chunks")
                        
                        # Create vector store
                        success = get_vector_store(text_chunks)
                        if success:
                            st.success("✅ Processing completed! You can now ask questions.")
                        else:
                            st.error("❌ Failed to create FAISS index.")
                            
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
            else:
                st.warning("Please upload at least one PDF file.")
                
if __name__ == "__main__":
    main()