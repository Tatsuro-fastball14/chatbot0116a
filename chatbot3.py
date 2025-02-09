from pathlib import Path
import sqlite3
import sys
import chromadb
import streamlit as st

from langchain import hub
from langchain.schema import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import openai

# Load environment variables
load_dotenv()

# Retrieve API key from Streamlit secrets or environment variables
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY が設定されていません。環境変数または .env ファイルを確認してください。")
    st.stop()

# Set the API key directly to the OpenAI client
openai.api_key = api_key
os.environ["OPENAI_API_KEY"] = api_key  # 環境変数としても設定

# Initialize ChatOpenAI with api_base
llm = ChatOpenAI(
    openai_api_key=api_key,
    model_name="gpt-4"
)

def initialize_vector_store() -> Chroma:
    """Initialize the VectorStore."""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
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

def initialize_chain():
    """Initialize the Langchain."""
    prompt = hub.pull("rlm/rag-prompt")
    retriever = initialize_retriever()

    def chain(user_input):
        try:
            retrieved_docs = retriever.get_relevant_documents(user_input)
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            formatted_prompt = prompt.format(context=context, question=user_input)
            response = llm.invoke(formatted_prompt)
            return response
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            return AIMessage(content="申し訳ありません、現在問題が発生しています。後でもう一度お試しください。")

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
        with st.spinner("GPTが入力中です..."):
            response = chain(user_input)
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


