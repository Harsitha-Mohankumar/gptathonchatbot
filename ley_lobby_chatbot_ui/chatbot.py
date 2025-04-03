import streamlit as st
import os
from langchain_community.vectorstores import FAISS
# from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

st.header("AI Chatbot")

st.cache_resource(show_spinner=False)
def load_model():
    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0 , convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    return model,embeddings

model,embeddings = load_model()

def get_more_relevant_docs(query,top_k):
    vector_store = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization= True)
    vector_store = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
    docs = vector_store.invoke(query)
    return docs

st.cache_resource(show_spinner=False)
def get_conversational_chain():
    
    system_prompt = """
You are AI ChatBot, an expert assistant specializing in providing detailed and accurate responses strictly based on retrieved information. Your goal is to deliver factual, concise, and helpful answers without introducing speculative or fabricated content.

*INSTRUCTIONS:*
- Base all responses exclusively on the provided context. If the information is not available, clearly state that you do not have enough data to answer.
- Avoid generating information that is not explicitly stated or implied by the retrieved documents.
- Respond politely and informatively.
- Use headings, bullet points, and concise paragraphs for clarity and readability.
- Highlight key points, participants, and outcomes. Avoid over-explaining or speculating beyond the given data.
- Emphasize important actions, follow-ups, and next steps from meetings or discussions.

*ONGOING CONVERSATION:*
The following is a record of the conversation so far, including user queries and assistant responses. Use this to maintain context and provide answers in continuity with previous exchanges.

{chat_history}

*DOCUMENT CONTEXT (if available):*
The following context is retrieved from relevant documents related to the query.

{context}

*USER QUERY:*
{input}

*ASSISTANT RESPONSE:*
Provide a detailed response, keeping prior exchanges in mind. Refer to past questions and answers for continuity. Avoid repeating information unnecessarily but expand on new aspects related to the user's follow-up query.
"""

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )
    
    # prompt  = hub.pull("langchain-ai/retrieval-qa-chat")
    # new_db = Chroma(persist_directory="chroma_db",embedding_function=embeddings)

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization= True)
    new_db = new_db.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    
    history_aware_retriever = create_history_aware_retriever(
    model, new_db, prompt
    )
    
    question_answer_chain = create_stuff_documents_chain(model, prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain )
    
    return rag_chain

chain = get_conversational_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.extend(
    [
        HumanMessage(content="hi there"),
        AIMessage(content="hi how can i help you?"),
    ]
    )

def user_input(user_question):
    
    global new_db,chain
        
    response = chain.invoke({"input": user_question, "context": "Your relevant context goes here" , "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend(
    [
        HumanMessage(content=user_question),
        AIMessage(content=response["answer"]),
    ]
    )
    
    return response

# Initialize chat history
if "messages_document" not in st.session_state:
    st.session_state.messages_document = []
    
# Display chat messages from history on app rerun
for message in st.session_state.messages_document:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        

def main():
    st.markdown('<div class="sidebar">', unsafe_allow_html=True)
    if prompt := st.chat_input("What is up?"):
        st.markdown('</div>', unsafe_allow_html=True)
        st.chat_message("user").markdown(prompt)
        st.session_state.messages_document.append({"role": "user", "content": prompt})
        
        with st.spinner('Wait for it...........'):  
            response = user_input(prompt)
            # revalatent_docs = response['context']
            # chat_history_info = response['chat_history']
            response = response['answer']
            st.chat_message("ai").markdown(response)
            
            with st.expander("see relevant documents"):
                revalatent_docs = get_more_relevant_docs(prompt,top_k=50)
                for doc in revalatent_docs:
                    st.write(doc)
                    st.markdown("""
                                
                                """)
                    
            # with st.expander("see chat history"):
            #     st.write(chat_history_info)
                          
        st.session_state.messages_document.append({"role": "assistant", "content": response})
        
if __name__ == "__main__":
    main()
