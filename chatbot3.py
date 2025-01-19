
import streamlit as st
from dotenv import load_dotenv
import os
from langchain import hub
from langchain.schema import AIMessage, HumanMessage
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# `.env` を明示的にロード
load_dotenv(dotenv_path=".env", override=True)

# OpenAI APIキーを取得
openai_api_key = os.getenv("OPENAI_API_KEY")

# 環境変数のチェック
if not openai_api_key:
    st.error("⚠ `.env` が正しく読み込まれていない可能性があります。環境変数を確認してください。")
    raise ValueError("`OPENAI_API_KEY` が設定されていません")

def initialize_vector_store() -> Chroma:
    """ベクトルストアを初期化"""
    embeddings = OpenAIEmbeddings()
    vector_store_path = "./resources/note.db"

    try:
        if Path(vector_store_path).exists():
            # 既存のデータベースをロード
            vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
        else:
            # `note.txt` の存在を確認
            note_path = Path("resources/note.txt")
            if not note_path.exists():
                raise FileNotFoundError("⚠ `resources/note.txt` が見つかりません")

            # テキストデータをロード
            loader = TextLoader(str(note_path))
            docs = loader.load()

            if not docs:
                raise ValueError("⚠ `resources/note.txt` にデータがありません")

            # テキストをチャンク分割
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = text_splitter.split_documents(docs)

            # ベクトルストアを作成
            vector_store = Chroma.from_documents(
                documents=documents, embedding=embeddings, persist_directory=vector_store_path
            )

        return vector_store

    except Exception as e:
        st.error(f"❌ ベクトルストアの初期化に失敗しました: {e}")
        raise e

def initialize_retriever() -> VectorStoreRetriever:
    """Retriever を初期化"""
    vector_store = initialize_vector_store()
    return vector_store.as_retriever()

def initialize_chain() -> RunnableSequence:
    """LangChain を初期化"""
    try:
        prompt = hub.pull("rlm/rag-prompt")  # プロンプトを取得
        llm = ChatOpenAI(api_key=openai_api_key)  # OpenAI APIキーを明示的に渡す
        retriever = initialize_retriever()

        chain = (
            {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
        )

        return chain
    except Exception as e:
        st.error(f"❌ Chain の初期化に失敗しました: {e}")
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
                    st.error(f"❌ エラーが発生しました: {e}")

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
        st.error(f"❌ アプリの実行中にエラーが発生しました: {e}")
        raise e

if __name__ == "__main__":
    main()
