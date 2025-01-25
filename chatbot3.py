from pathlib import Path
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
import streamlit as st

from langchain import hub
from langchain.schema import AIMessage, HumanMessage
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai
from dotenv import load_dotenv
import os
from openai import OpenAI
from chromadb import PersistentClient
from langchain.docstore.document import Document

def initialize_vector_store() -> Chroma:
    """Initialize the VectorStore."""
    embeddings = OpenAIEmbeddings()

    vector_store_path = "./resources/note.db"
    if Path(vector_store_path).exists():
        vector_store = Chroma(embedding_function=embeddings, persist_directory=vector_store_path)
    else:
        loader = TextLoader("resources/note.txt")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        vector_store = Chroma.from_documents(
            documents=splits, embedding=embeddings, persist_directory=vector_store_path
        )

    return vector_store

def initialize_retriever() -> VectorStoreRetriever:
    """Initialize the Retriever."""
    vector_store = initialize_vector_store()
    return vector_store.as_retriever()

def initialize_chain() -> RunnableSequence:
    """Initialize the Langchain."""
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI()
    retriever = initialize_retriever()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    )
    return chain

def main() -> None:
    """Main function for the ChatGPT using Streamlit."""
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
            response = chain.invoke(user_input)
        st.session_state.messages.append(AIMessage(content=response.content))

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            st.write(f"System message: {message.content}")

if __name__ == "__main__":
    main()
