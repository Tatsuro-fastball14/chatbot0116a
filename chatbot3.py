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

# 環境変数を読み込む
load_dotenv()

def initialize_vector_store() -> Chroma:
    """ベクトルストアを初期化"""
    embeddings = OpenAIEmbeddings()
    vector_store_path = "./resources/note.db"

    try:
        if Path(vector_store_path).exists():
            # 既存のデータベースを読み込む
            vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
        else:
            # 新しいベクトルストアを作成
            loader = TextLoader("resources/note.txt")
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            vector_store = Chroma.from_documents(
                documents=splits, embedding=embeddings, persist_directory=vector_store_path
            )

        return vector_store
    except Exception as e:
        st.error(f"ベクトルストアの初期化に失敗しました: {e}")
        raise e  # 例外を再スローして詳細を確認

def initialize_retriever() -> VectorStoreRetriever:
    """Retriever を初期化"""
    vector_store = initialize_vector_store()
    return vector_store.as_retriever()

def initialize_chain() -> RunnableSequence:
    """LangChain を初期化"""
    try:
        prompt = hub.pull("rlm/rag-prompt")  # プロンプトを取得
        llm = ChatOpenAI()
        retriever = initialize_retriever()

        chain = (
            {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
        )

        return chain
    except Exception as e:
        st.error(f"Chain の初期化に失敗しました: {e}")
        raise e

def main() -> None:
    """Streamlit アプリのメイン関数"""
    try:
        chain = initialize_chain()

        # ページ設定
        st.set_page_config(page_title="RAG ChatGPT")
        st.header("RAG ChatGPT")

        # チャット履歴の管理
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # ユーザー入力の監視
        user_input = st.chat_input("聞きたいことを入力してね！")

        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("GPT is typing ..."):
                try:
                    response = chain.invoke(user_input)
                    st.session_state.messages.append(AIMessage(content=response.content))
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

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

    except Exception as e:
        st.error(f"アプリの実行中にエラーが発生しました: {e}")
        raise e

if __name__ == "__main__":
    main()
