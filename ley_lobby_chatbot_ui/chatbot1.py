import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load CSS
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

@st.cache_resource(show_spinner=False)
def load_model():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                             temperature=0.8, convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return model, embeddings

try:
    model, embeddings = load_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

def analyze_text(text):
    """Perform comprehensive text analysis"""
    if not text or not isinstance(text, str):
        return {}
        
    analysis = {}
    
    # Basic stats
    words = re.findall(r'\w+', text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    analysis['word_count'] = len(words)
    analysis['sentence_count'] = len(sentences)
    analysis['paragraph_count'] = len(paragraphs)
    analysis['avg_word_length'] = sum(len(word) for word in words)/len(words) if words else 0
    analysis['avg_sentence_length'] = sum(len(s.split()) for s in sentences)/len(sentences) if sentences else 0
    
    # Vocabulary analysis
    vocab = set(words)
    analysis['vocab_size'] = len(vocab)
    analysis['lexical_diversity'] = len(vocab)/len(words) if words else 0
    
    # Keyword extraction (top 20)
    stopwords = set(['the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 'it'])
    meaningful_words = [word for word in words if word not in stopwords and len(word) > 3]
    word_freq = Counter(meaningful_words)
    analysis['top_keywords'] = word_freq.most_common(20)
    
    return analysis

def visualize_analysis(analysis):
    """Create visualizations for the text analysis"""
    if not analysis:
        st.warning("No analysis data to visualize")
        return
        
    st.subheader("Document Analysis Dashboard")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Words", analysis.get('word_count', 0))
    with col2:
        st.metric("Unique Words", analysis.get('vocab_size', 0))
    with col3:
        st.metric("Lexical Diversity", f"{analysis.get('lexical_diversity', 0):.2%}")
    
    # Keyword visualization
    st.subheader("Top Keywords")
    keywords = dict(analysis.get('top_keywords', []))
    
    if keywords:
        fig, ax = plt.subplots()
        ax.barh(list(keywords.keys())[::-1], list(keywords.values())[::-1])
        ax.set_xlabel('Frequency')
        st.pyplot(fig)
        
        # Word cloud
        st.subheader("Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keywords)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("No keywords to display")

def process_uploaded_file(uploaded_file):
    """Process different file types and extract text"""
    if not uploaded_file:
        return None
        
    text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        if uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(tmp_file_path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif uploaded_file.name.endswith('.txt'):
            with open(tmp_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(tmp_file_path)
            text = df.to_string()
        else:
            st.error("Unsupported file format")
            return None
    except Exception as e:
        st.error(f"Error processing file: {e}")
        logger.error(f"Error processing file: {str(e)}")
        return None
    finally:
        try:
            os.unlink(tmp_file_path)
        except:
            pass
    
    return text if text.strip() else None

@st.cache_resource
def create_vector_store(text):
    """Create FAISS vector store from text"""
    if not text:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    """Create the conversation chain with proper context handling"""
    if not vector_store:
        return None
        
    # System prompt with context placeholder
    system_prompt = """
    You are an expert document analyst. Use the following context to answer the question.
    Context: {context}
    
    Important Instructions:
    - Answer based only on the provided context
    - Be detailed and thorough
    - Use bullet points when appropriate
    - If the answer isn't in the context, say so
    """
    
    # Create prompt template that includes context
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Create the full chain
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return retrieval_chain

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

uploaded_file = st.file_uploader("Upload your document (PDF, TXT, CSV)", 
                                type=['pdf', 'txt', 'csv'])

if uploaded_file is not None:
    with st.spinner("Processing your document..."):
        text = process_uploaded_file(uploaded_file)
        
        if text:
            # Document analysis section
            analysis = analyze_text(text)
            visualize_analysis(analysis)
            
            # Create vector store
            vector_store = create_vector_store(text)
            
            # Update session state
            st.session_state.chat_history = []
            st.session_state.vector_store = vector_store
            st.session_state.document_processed = True
            
            st.success("Document processed successfully! You can now chat with it.")

# Chat interface (only show if document is processed)
if st.session_state.get('document_processed', False) and st.session_state.vector_store:
    st.subheader("Chat with Your Document")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
    
    # Chat input
    if prompt := st.chat_input("Ask about the document..."):
        # Add user message to chat history and display immediately
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                try:
                    chain = get_conversational_chain(st.session_state.vector_store)
                    if not chain:
                        st.error("Failed to create conversation chain")
                    else:
                        response = chain.invoke({
                            "input": prompt,
                            "chat_history": st.session_state.chat_history
                        }, config=RunnableConfig(timeout=15))
                        
                        # Display and store the response
                        st.markdown(response["answer"])
                        st.session_state.chat_history.append(AIMessage(content=response["answer"]))
                        
                        # Show relevant passages
                        with st.expander("View relevant passages"):
                            for i, doc in enumerate(response["context"], 1):
                                st.write(f"**Passage {i}:**")
                                st.write(doc.page_content)
                                st.divider()
                
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    logger.error(f"Error generating response: {str(e)}")
                    if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], HumanMessage):
                        st.session_state.chat_history.pop()  # Remove the last user message if failed
st.markdown('</div>', unsafe_allow_html=True)