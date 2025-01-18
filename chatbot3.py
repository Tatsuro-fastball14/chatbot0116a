from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.schema import AIMessage, HumanMessage
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os

# Load environment variables
load_dotenv()

def initialize_vector_store() -> Chroma:
    """Initialize the VectorStore."""
    embeddings = OpenAIEmbeddings()

    vector_store_path = "./resources/note.db"
    if Path(vector_store_path).exists():
        # Load the persistent vector store
        vector_store = Chroma(embedding_function=embeddings, persist_directory=vector_store_path)
    else:
        # Load documents and create a new vector store
        try:
            loader = TextLoader("resources/note.txt")
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            vector_store = Chroma.from_documents(
                documents=splits, embedding=embeddings, persist_directory=vector_store_path
            )
        except Exception as e:
            st.error(f"Error initializing vector store: {e}")
            raise e

    return vector_store

def initialize_retriever() -> VectorStoreRetriever:
    """Initialize the Retriever."""
    vector_store = initialize_vector_store()
    return vector_store.as_retriever()

def initialize_chain() -> RunnableSequence:
    """Initialize the LangChain."""
    prompt = hub.pull("rlm/rag-prompt")  # Ensure this prompt is available
    llm = ChatOpenAI()
    retriever = initialize_retriever()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    )
    return chain

def main() -> None:
    """Main function for the ChatGPT using Streamlit."""
    try:
        chain = initialize_chain()

        # Configure the page
        st.set_page_config(page_title="RAG ChatGPT")
        st.header("RAG ChatGPT")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Monitor user input
        if user_input := st.chat_input("聞きたいことを入力してね！"):
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("GPT is typing ..."):
                response = chain.invoke({"context": st.session_state.messages, "question": user_input})
            st.session_state
