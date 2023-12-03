import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OllamaEmbeddings
import os
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def main():
    # Create two tabs
    tabs = ["Home", "PDF Summerizer", "PDF Chat"]
    selected_tab = st.sidebar.radio("Select a tab", tabs, key="tabs")

    if selected_tab == "Home":
        home()
    elif selected_tab == "PDF Summerizer":
        summary()
    elif selected_tab == "PDF Chat":
        chat()

# Sidebar contents
with st.sidebar:
    st.title('Welcome to ConCall Summeriser!')
    add_vertical_space(5)

load_dotenv()

def home():
    if "VectorStore" not in st.session_state:
        st.session_state.VectorStore = None

    st.header("Upload a PDF to chat with")
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                st.session_state.VectorStore = pickle.load(f)
        else:
            with st.spinner(text="Vectorizing PDF..."):
                embeddings = OllamaEmbeddings(model="mistral")
                st.session_state.VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(st.session_state.VectorStore, f)

def summary():
    st.header("PDF Summerizer")
    
    
     # Accept user questions/query
    query = st.text_input("Ask questions about your PDF file:")
    if query and st.session_state.VectorStore:
        docs = st.session_state.VectorStore.similarity_search(query=query, k=3)

        llm = Ollama(
            model="mistral",
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        chain = load_qa_chain(llm=llm, chain_type="stuff")
  
        response = chain.run(input_documents=docs, question=query)
         

        # Store the chat results in session_state
        if "ChatResults" not in st.session_state:
            st.session_state.ChatResults = []
        
        st.session_state.ChatResults.append(response)

        # Display chat results
        for result in st.session_state.ChatResults:
            st.write(result)

def chat():
    st.header("Chat with PDF 💬")

    # Initialize session state if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    # Check if a new chat has been requested
    if st.button("New Chat"):
        # Get the current conversation
        current_conversation = st.session_state.get("current_conversation", [])
        
        # Store the current conversation in the list
        current_conversation.append(st.session_state.messages)
        
        # Keep only the last 5 conversations
        if len(current_conversation) > 5:
            current_conversation.pop(0)
        
        # Reset the conversation state
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        
        # Update the current conversation list
        st.session_state.current_conversation = current_conversation

    chat = Ollama(
        model="mistral",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

    chain = load_qa_chain(llm=chat, chain_type="stuff")

    # Accumulate the entire conversation history
    all_messages = st.session_state.get("current_conversation", []) + st.session_state.get('messages', [])

    # user input directly on the main page
    user_input = st.text_input("Your message:", key="user_input")
    
    try:
        docs = st.session_state.VectorStore.similarity_search(query=user_input, k=3)
    except AttributeError as e:
        st.write("!!! Document Not Loaded Yet !!!")

    # handle user input
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking..."):
            response = chain.run(input_documents=docs, question=user_input, context=all_messages)
        st.session_state.messages.append(
                AIMessage(content=response))

    # Update the conversation history
    st.session_state.messages = all_messages

    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')


if __name__ == '__main__':
    main()
