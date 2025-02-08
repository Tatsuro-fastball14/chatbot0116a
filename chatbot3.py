from pathlib import Path
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
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

# 環境変数をロード
load_dotenv()

# APIキーを取得
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("openai", {}).get("api_key")
api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")  # API Base を環境変数から取得

if not api_key:
    st.error("OPENAI_API_KEY が設定されていません。環境変数または .env ファイルを確認してください。")
    st.stop()

# APIキーを環境変数と OpenAI クライアントに設定
openai.api_key = api_key
os.environ["OPENAI_API_KEY"] = api_key

# ChatOpenAI を初期化
try:
    llm = ChatOpenAI(
        api_key=api_key,
        model_name="gpt-4",  # または "gpt-3.5-turbo"
        model_kwargs={"api_base": api_base}  # 修正
    )
except Exception as e:
    st.error(f"ChatOpenAI の初期化に失敗しました: {e}")
    st.stop()

def initialize_vector_store() -> Chroma:
    """VectorStore を初期化"""
    embeddings = OpenAIEmbeddings(api_key=api_key)
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

def initialize_chain():
    """LangChain の初期化"""
    prompt = hub.pull("rlm/rag-prompt")
    retriever = initialize_retriever()

    def chain(user_input):
        try:
            retrieved_docs = retriever.invoke(user_input)  # 修正
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            formatted_prompt = prompt.format(context=context, question=user_input)
            response = llm.invoke(formatted_prompt)
            return response
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            return AIMessage(content="申し訳ありません、現在問題が発生しています。後でもう一度お試しください。")

    return chain

def main() -> None:
    """Streamlit 用の ChatGPT アプリケーションのメイン関数"""
    st.set_page_config(page_title="RAG ChatGPT")  # ✅ 最初に実行
    
    st.write("API Key (masked):", api_key[:10] + "********")  # APIキーが正しく取得されているか確認
    st.write("API Base URL:", api_base)  # API Base URL 確認
    
    chain = initialize_chain()

    # ページ設定
    st.header("RAG ChatGPT")

    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ユーザー入力の監視
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("GPTが入力中です..."):
            response = chain(user_input)
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

