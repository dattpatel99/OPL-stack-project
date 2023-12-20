import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from multiprocessing import Pool
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
import pinecone
import os
from template import bot_template, css, user_template

def get_pdf_text(pdf_doc):
    text = ""
    reader = PdfReader(pdf_doc)
    for _ in reader.pages:
            text += _.extract_text()
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator='\n\n', chunk_size=1000, chunk_overlap = 200, length_function=len)
    chunk = splitter.split_text(text)
    return chunk
     
def get_vectorstores(chunks):
    pinecone.init(
    api_key= os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get("PINECONE_ENV"))
    index_name = os.environ.get("PINECONE_INDEX")
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    pinecone.create_index(name=index_name, metric="cosine", shards=1, dimension=1536)
    embeddings = OpenAIEmbeddings()
    vectorstores = Pinecone.from_texts(chunks, embeddings,index_name=index_name) 
    return vectorstores

def get_chain_conversation(store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=store,
        memory=memory
    )
    return conversation_chain

def handle_query(query):
    response = st.session_state.conversation({'question': query})
    st.session_state.chat_history = response['chat_history']
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}",msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",msg.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title ="Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books")
    query = st.text_input("Ask a question about your documents:")
    if query:
        handle_query(query)

    with st.sidebar:
        st.subheader("Upload your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here, press 'Process'", type=['pdf'], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                with Pool(5) as p:
                    texts = '\n\n'.join(p.map(get_pdf_text, pdf_docs))                    
                # data chunks
                chunk = get_text_chunks(texts)
                # vector storage
                vectorestores =  get_vectorstores(chunk)
                # convo chain
                st.session_state.conversation = get_chain_conversation(vectorestores.as_retriever())

if __name__ == '__main__':
    main()