import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile
import base64

# Set page config must be the first Streamlit command
st.set_page_config("Data Insight Generator", page_icon="📊")

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Check and install missing dependencies
try:
    import openpyxl
except ImportError:
    st.warning("The 'openpyxl' package is required for Excel file support. Installing now...")
    os.system("pip install openpyxl")
    import openpyxl  # Try importing again after installation

# List of alternative model names
MODEL_NAMES = [
    "gemini-1.0-pro",
    "gemini-pro",
    "models/gemini-pro",
    "gemini-1.5-pro",
    "gemini-1.0-pro-001"
]
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file {file_name} not found. Using default styles.")

load_css("style.css")

# Add navbar at the top
st.markdown(
    """
    <style>
        body, p, h1, h2, h3, h4, h5, h6, div, span, a {
            color: black !important;
        }
        .navbar {
            position: fixed;
            top: 35px;
            left: 0;
            width: 100%;
            background-color: #8A2BE2 !important;
            padding: 15px 0;
            z-index: 1000;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .navbar-title {
            
            font-size: 24px;
            font-weight: bold;
            text-align: left;
            margin-left: 20px;
            padding: 5px 0;
        }
        .stApp {
            padding-top: 100px !important;
        }
    </style>
    
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64.b64encode(open('image.png', 'rb').read()).decode()}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        
        background-attachment: fixed;
    }}
    .main-content {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        margin-top: 60px;
    }}
    
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="main-content">', unsafe_allow_html=True)

def extract_text_from_pdf(pdf_docs):
    """Extract text from PDF files"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Only add if text was extracted
                text += page_text + "\n"
    return text if text else None

def extract_text_from_data(file):
    """Extract text from CSV/Excel files"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        # Read based on file type
        if file.name.endswith('.csv'):
            df = pd.read_csv(tmp_path)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(tmp_path, engine='openpyxl')
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Convert dataframe to text representation with metadata
        text = f"FILE: {file.name}\n"
        text += f"COLUMNS: {', '.join(df.columns)}\n"
        text += f"SAMPLE DATA:\n{df.head(5).to_string(index=False)}\n"
        text += f"DATA SUMMARY:\n{df.describe().to_string()}"
        return text
        
    except Exception as e:
        st.error(f"Error reading {file.name}: {str(e)}")
        return None

def get_text_chunks(text):
    """Split text into chunks"""
    if not text:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save vector store"""
    if not text_chunks:
        raise ValueError("No valid text chunks to process")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create conversation chain with model fallback"""
    prompt_template = """
    You are an expert data analyst and content interpreter. 
    Analyze the provided context and answer the question thoroughly.
    
    For tabular data:
    - Identify relevant columns for the question
    - Perform calculations if needed
    - Highlight trends and patterns
    - Provide statistical insights when appropriate
    
    For document content:
    - Extract key information
    - Summarize when appropriate
    - Maintain original meaning
    
    If the answer isn't in the context, say:
    "I couldn't find this information in the provided data."
    
    Context:\n{context}\n
    Question: \n{question}\n

    Answer:
    """
    
    for model_name in MODEL_NAMES:
        try:
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            model.invoke("Test")  # Test connection
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            return chain
            
        except Exception as e:
            print(f"Model {model_name} failed, trying next... Error: {str(e)}")
            continue
    
    raise Exception("None of the model names worked. Please check your API access.")

def generate_response(user_question, data_type):
    """Handle user question and generate appropriate response"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        return response["output_text"]
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    """Main Streamlit app"""
    st.header("Chat with Your Documents & Data")
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'data_type' not in st.session_state:
        st.session_state.data_type = None
    
    # File upload section
    with st.sidebar:
        st.title("Upload Files")
        file_type = st.radio(
            "Select file type:",
            ("PDF Documents", "CSV/Excel Data"),
            index=0
        )
        
        uploaded_files = st.file_uploader(
            f"Upload your {'PDF' if file_type == 'PDF Documents' else 'CSV/Excel'} files",
            accept_multiple_files=True,
            type=["pdf"] if file_type == "PDF Documents" else ["csv", "xlsx", "xls"]
        )
        
        if st.button("Process Files"):
            if not uploaded_files:
                st.warning("Please upload at least one file")
            else:
                with st.spinner("Processing files..."):
                    try:
                        all_text = ""
                        processed_count = 0
                        
                        for file in uploaded_files:
                            if file_type == "PDF Documents":
                                file_text = extract_text_from_pdf([file])
                            else:
                                file_text = extract_text_from_data(file)
                            
                            if file_text:
                                all_text += f"\n\n--- {file.name} ---\n{file_text}"
                                processed_count += 1
                        
                        if processed_count > 0:
                            text_chunks = get_text_chunks(all_text)
                            if text_chunks:
                                get_vector_store(text_chunks)
                                st.session_state.processed = True
                                st.session_state.data_type = "document" if file_type == "PDF Documents" else "tabular"
                                st.success(f"Successfully processed {processed_count}/{len(uploaded_files)} files!")
                            else:
                                st.error("No readable content found in the processed files")
                        else:
                            st.error("None of the files could be processed")
                            
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
    
    # Query section
    if st.session_state.processed:
        st.subheader("Ask Questions About Your Data")
        user_question = st.text_area(
            "Enter your question here", 
            height=100,
            placeholder="What insights can you provide about this data?"
        )
        
        if st.button("Get Answer") and user_question:
            with st.spinner("Analyzing your data..."):
                response = generate_response(user_question, st.session_state.data_type)
                st.subheader("Analysis Results")
                st.write(response)
                
                # Show sample data for tabular queries
                if st.session_state.data_type == "tabular" and uploaded_files:
                    with st.expander("View Sample Data from First File"):
                        try:
                            sample_file = uploaded_files[0]
                            if sample_file.name.endswith('.csv'):
                                df = pd.read_csv(sample_file)
                            else:
                                df = pd.read_excel(sample_file, engine='openpyxl')
                            st.dataframe(df.head(5))
                            st.write(f"Shape: {df.shape} (rows × columns)")
                        except Exception as e:
                            st.error(f"Couldn't display sample data: {str(e)}")

if __name__ == "__main__":
    main()