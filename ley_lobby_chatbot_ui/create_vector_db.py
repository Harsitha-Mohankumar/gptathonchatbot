from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import google.generativeai as genai
import os  

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1 , separators=["---"])
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # vector_store = Chroma.from_texts(text_chunks, embedding = embeddings , persist_directory="chroma_db")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_Ministry_of_Health")
    
if __name__ == "__main__":
    text = get_pdf_text(r"enhanced_meeting_details_story(Ministry of Health Translated).txt")
    print("Reading text from file completed successfully......................")
    chunks = get_text_chunks(text)
    print("Text chunks created successfully......................")
    get_vector_store(chunks)
    print("Vector store created successfully......................")