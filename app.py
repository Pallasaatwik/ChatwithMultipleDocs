import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
import shutil
import stat
from io import BytesIO
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
import gc

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Describe image with Gemini Vision ---
def describe_image_with_gemini_pil(pil_image, page_num, img_num):
    """Enhanced image description with better context"""
    try:
        img_bytes = BytesIO()
        pil_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Enhanced prompt for better image analysis
        prompt = """Describe this image in detail. Include:
        - What type of image it is (chart, diagram, photo, illustration, etc.)
        - Key visual elements, text, numbers, or data visible
        - Any relationships, patterns, or important information
        - Context that might be relevant for understanding the document
        Be specific and comprehensive."""
        
        response = model.generate_content([prompt, pil_image])
        return response.text
    except Exception as e:
        st.warning(f"Image description failed for Page {page_num}, Image {img_num}: {e}")
        return f"[Image description failed: {e}]"

# --- Extract text + image descriptions from PDF ---
def get_pdf_text_and_images(pdf_docs):
    text = ""
    total_images = 0
    
    for pdf_idx, pdf in enumerate(pdf_docs):
        st.info(f"Processing PDF {pdf_idx + 1}/{len(pdf_docs)}: {pdf.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            temp_pdf_path = tmp_file.name

        try:
            doc = fitz.open(temp_pdf_path)
            
            for page_index in range(len(doc)):
                page = doc[page_index]
                
                # Extract text
                page_text = page.get_text()
                if page_text.strip():
                    text += f"\n--- Page {page_index+1} Text ---\n{page_text}\n"
                
                # Extract images
                images = page.get_images(full=True)
                if images:
                    st.info(f"Processing {len(images)} images on page {page_index+1}")
                
                for img_index, img in enumerate(images, start=1):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        
                        
                        pil_image = Image.open(BytesIO(img_bytes))
                        
                        
                        if pil_image.size[0] < 50 or pil_image.size[1] < 50:
                            continue
                            
                        description = describe_image_with_gemini_pil(pil_image, page_index+1, img_index)
                        text += f"\n--- Image on Page {page_index+1} (Image {img_index}) ---\n{description}\n"
                        total_images += 1
                        
                    except Exception as e:
                        st.warning(f"Failed to process image {img_index} on page {page_index+1}: {e}")
                        text += f"\n--- Image on Page {page_index+1} (Image {img_index}) ---\n[Image extraction failed: {e}]\n"
            
            doc.close()
            
        except Exception as e:
            st.error(f"Failed to process PDF {pdf.name}: {e}")
        finally:
            try:
                os.unlink(temp_pdf_path)
            except Exception:
                pass
    
    st.success(f"Extraction complete! Processed {total_images} images total.")
    return text

# --- Split into chunks ---
def get_text_chunks(text):
    if not text.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000,
        separators=["\n--- Page", "\n--- Image", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    st.info(f"Created {len(chunks)} text chunks")
    return chunks

# --- Create FAISS index ---
def get_vector_store(text_chunks):
    def remove_readonly(func, path, _):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    gc.collect()

    # Clean up existing index
    if os.path.exists("faiss_index"):
        try:
            shutil.rmtree("faiss_index", onerror=remove_readonly)
        except PermissionError:
            temp_name = f"faiss_index_old_{os.getpid()}"
            os.rename("faiss_index", temp_name)
            try:
                shutil.rmtree(temp_name, onerror=remove_readonly)
            except Exception:
                pass

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        os.makedirs("faiss_index", exist_ok=True)
        vector_store.save_local("faiss_index")
        st.success(f"✅ FAISS index created successfully with {len(text_chunks)} chunks!")
    except Exception as e:
        st.error(f"Failed to create FAISS index: {e}")
        raise

# --- Generate answer using Gemini directly ---
def generate_answer(context, question):
    """Use Gemini directly to generate answers"""
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
        prompt = f"""
        You are an AI assistant that can answer questions about PDF documents that contain both text and images.
        The context below includes both text content and detailed descriptions of images from the PDF.
        
        Answer the question as detailed as possible from the provided context.
        When referring to images, mention the page number and image details.
        If the answer is not in the provided context, say: "Answer is not available in the context".
        Don't guess or make up information.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        
        response = model.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error generating answer: {e}"

# --- Handle user query ---
def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.error("❌ No FAISS index found. Please upload and process PDF files first.")
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=5)  # Get top 5 relevant chunks

        if not docs:
            st.warning("No relevant content found for your question.")
            return

        # Combine context from all relevant documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        with st.spinner("Generating answer..."):
            answer = generate_answer(context, user_question)

        st.write("**Reply:**", answer)
        
        # Show source information
        with st.expander("📖 Source Context"):
            for i, doc in enumerate(docs):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                st.write("---")
                
    except Exception as e:
        st.error(f"Error processing your question: {e}")
        st.error(f"Error details: {str(e)}")

# --- Main app ---
def main():
    st.set_page_config(
        page_title="Chat with PDF (Text + Images)",
        page_icon="📄",
        layout="wide"
    )
    st.header("📄 Chat with PDF (Text + Image Analysis) using Gemini 💁")

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
        return

    # Status check
    if os.path.exists("faiss_index"):
        st.success("✅ FAISS index is ready! You can ask questions about your PDFs.")
    else:
        st.warning("⚠️ Please upload and process PDF files first.")

    # Main chat interface
    user_question = st.text_input("Ask a Question from the PDF Files (including about images)")
    
    if user_question:
        user_input(user_question)

    # Sidebar for file upload
    with st.sidebar:
        st.title("📁 Menu:")
        
        pdf_docs = st.file_uploader(
            "Upload your PDF Files", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("🚀 Submit & Process"):
            if pdf_docs:
                try:
                    with st.spinner("Processing PDFs and analyzing images..."):
                        raw_text = get_pdf_text_and_images(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("❌ No text or image descriptions found in the uploaded PDF files.")
                            return
                        
                        text_chunks = get_text_chunks(raw_text)
                        
                        if not text_chunks:
                            st.error("❌ No text chunks created.")
                            return
                        
                        get_vector_store(text_chunks)
                        st.success("✅ Processing completed! You can now ask questions about text and images.")
                        st.rerun()  # Refresh the app to show updated status
                        
                except Exception as e:
                    st.error(f"❌ Error during processing: {e}")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")
        
        # Help section
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Upload PDFs**: Choose one or more PDF files
            2. **Process**: Click 'Submit & Process' to extract text and analyze images
            3. **Ask Questions**: You can ask about:
               - Text content from any page
               - Images and their descriptions
               - Charts, diagrams, or visual elements
               - Relationships between text and images
            
            **Example questions:**
            - "What does the chart on page 3 show?"
            - "Describe the images in this document"
            - "What information is in the diagram?"
            """)

if __name__ == "__main__":
    main()