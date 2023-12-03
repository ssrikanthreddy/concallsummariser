import streamlit as st
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
 
def main():
 
    # Create two tabs
    tabs = ["Home", "PDF Summerizer", "PDF Chat"]
    selected_tab = st.sidebar.radio("Select a tab", tabs)

    if selected_tab == "Home":
        home()
    elif selected_tab == "PDF Summerizer":
        summary()
    elif selected_tab == "PDF Chat":
        chat()

# Sidebar contents
with st.sidebar:
    st.title('Chat App')
 
    add_vertical_space(5)
    
 
load_dotenv()
 

    
    
 
def home():
    st.header("Upload a PDF to chat with")
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
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
 
        # # embeddings
        # store_name = pdf.name[:-4]
        store_name = "wateract"
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            #   st.write('Embeddings Loaded from the Disk')
        else:
            embeddings = OllamaEmbeddings(model="mistral")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
       
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # query = "summerise the document"
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = Ollama(  model="mistral",
                    #model='Llama2',
                    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
 

def summary():
    st.header("PDF Summerizer")
    pass


def chat():
    st.header("Chat with PDF 💬")
    pass


if __name__ == '__main__':
    main()