from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.schema import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai
import os
import chromadb

print(chromadb.__version__)

# 環境変数をロード
load_dotenv()

# APIキーを取得
auth_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("openai", {}).get("api_key")
api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

if not auth_key:
    st.error("OPENAI_API_KEY が設定されていません。環境変数または .env ファイルを確認してください。")
    st.stop()

openai.api_key = auth_key
os.environ["OPENAI_API_KEY"] = auth_key

def initialize_vector_store() -> Chroma:
    """VectorStore を初期化"""
    embeddings = OpenAIEmbeddings(api_key=auth_key)
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
    """Retriever を初期化"""
    vector_store = initialize_vector_store()
    return vector_store.as_retriever()

def initialize_chain() -> RunnableSequence:
    """Langchain を初期化"""
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(api_key=auth_key, model_name="gpt-4", model_kwargs={"api_base": api_base})
    retriever = initialize_retriever()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    )
    return chain

def main() -> None:
    """Streamlit 用の ChatGPT アプリケーションのメイン関数"""
    st.set_page_config(page_title="RAG ChatGPT")
    st.header("RAG ChatGPT")
    
    chain = initialize_chain()
    
    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # ユーザー入力の監視
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("GPT is typing ..."):
            response = chain.invoke(user_input)
        st.session_state.messages.append(AIMessage(content=response.content))
    
    # チャット履歴の表示
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
