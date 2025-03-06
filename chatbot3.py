from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from langchain import hub
from langchain.schema import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai
from streamlit_chat import message
import os

# ✅ OpenAI APIキーの取得
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("openai", {}).get("api_key")

if not api_key:
    st.error("OPENAI_API_KEY が設定されていません。環境変数または .env ファイルを確認してください。")
    st.stop()

openai.api_key = api_key

# ✅ ChatOpenAIの初期化
llm = ChatOpenAI(
    openai_api_key=api_key,
    model_name="gpt-4",
)

# ✅ ログイン処理
def load_config():
    """config.yaml を読み込む"""
    config_path = "config.yaml"
    if not Path(config_path).exists():
        st.error(f"{config_path} が見つかりません。")
        st.stop()
    with open(config_path) as file:
        return yaml.load(file, Loader=SafeLoader)

def login_user():
    """ユーザーのログイン処理"""
    config = load_config()

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    try:
        authenticator.login(key="auth_login")  # ✅ `key` を追加
        if st.session_state.get("authentication_status"):
            with st.sidebar:
                st.markdown(f'## ようこそ *{st.session_state["name"]}*')
                authenticator.logout('ログアウト', 'sidebar', key="logout_button")  # ✅ ログアウトにも `key` を追加
                st.divider()
            st.session_state["logged_in"] = True
            return True
        elif st.session_state.get("authentication_status") is False:
            st.error('ユーザー名またはパスワードが間違っています')
        elif st.session_state.get("authentication_status") is None:
            st.warning('ユーザー名とパスワードを入力してください')
    except Exception as e:
        st.error(f"ログインエラー: {e}")

    st.session_state["logged_in"] = False
    return False

# ✅ LangChain の初期化
def initialize_vector_store() -> Chroma:
    """Initialize the VectorStore."""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store_path = "./resources/note.db"

    if Path(vector_store_path).exists():
        vector_store = Chroma(embedding_function=embeddings, persist_directory=vector_store_path)
    else:
        loader = TextLoader("resources/note.txt", encoding='utf-8')
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        vector_store = Chroma.from_documents(
            documents=splits, embedding=embeddings, persist_directory=vector_store_path
        )

    return vector_store

def initialize_retriever() -> VectorStoreRetriever:
    """Retrieverを初期化します。"""
    vector_store = initialize_vector_store()
    return vector_store.as_retriever()

def initialize_chain():
    """LangChainを初期化します。"""
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

# ✅ メイン関数
def main() -> None:
    """メイン関数"""

    # ✅ ログイン処理
    login_status = login_user()

    if not login_status:
        st.stop()  # 未ログインならストップ

    # ✅ サイドバーメニュー
    menu = ["ホーム", "ヘルプ"]
    choice = st.sidebar.selectbox("メニュー", menu, key="menu_select")

    if choice == "ホーム":
        st.title("🤖 AI チャットボット")

        chain = initialize_chain()

        # チャット履歴の初期化
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # ユーザー入力の監視
        user_input = st.text_input("聞きたいことを入力してください:", key="user_input")  # ✅ `key="user_input"` を追加
        if user_input:
            st.session_state.messages.append({"role": "user", "content": HumanMessage(content=user_input)})
            with st.spinner("GPTが入力中です..."):
                ai_response = chain(user_input)

            st.session_state.messages.append({"role": "assistant", "content": AIMessage(content=ai_response.content)})

        # メッセージの表示
        for i, msg in enumerate(st.session_state.messages):
            if msg['role'] == 'user':
                message(msg['content'].content, is_user=True, avatar_style="personas", key=f"user_{i}")
            elif msg['role'] == 'assistant':
                message(msg['content'].content, is_user=False, avatar_style="bottts", key=f"assistant_{i}")

    elif choice == "ヘルプ":
        st.title("ヘルプ")
        st.write("ここにヘルプ情報を記載します。")

if __name__ == "__main__":
    main()
