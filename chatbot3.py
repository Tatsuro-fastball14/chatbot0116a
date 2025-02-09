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
import openai  # 追加

# 環境変数をロード
load_dotenv()

# APIキーを安全に取得
api_key = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY が設定されていません。環境変数または .env ファイルを確認してください。")
    st.stop()

# OpenAIのAPIキーをセット
openai.api_key = api_key
os.environ["OPENAI_API_KEY"] = api_key  # 環境変数としても設定

# ChatOpenAI の初期化
llm = ChatOpenAI(
    openai_api_key=api_key,
    model_name="gpt-4",  # または "gpt-3.5-turbo"
)

def initialize_vector_store() -> Chroma:
    """ベクトルストアの初期化"""
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
    """Retriever の初期化"""
    vector_store = initialize_vector_store()
    return vector_store.as_retriever()

def initialize_chain():
    """Langchainの初期化"""
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
    """Streamlit を使った ChatGPT アプリのメイン関数"""
    chain = initialize_chain()

    # ページ設定
    st.set_page_config(page_title="RAG ChatGPT")
    st.header("RAG ChatGPT")

    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ユーザーの入力を取得
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("GPTが入力中です..."):
            response = chain(user_input)
        st.session_state.messages.append(AIMessage(content=response.content))

    # チャット履歴を表示
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




