from pathlib import Path
import streamlit as st
from langchain import hub
from langchain.schema import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai
from pathlib import Path
from streamlit_chat import message
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Pre-hashing all plain text passwords once
# stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)
try:
    authenticator.login()
    if st.session_state["authentication_status"]:
        ## ログイン成功
        with st.sidebar:
            st.markdown(f'## Welcome *{st.session_state["name"]}*')
            authenticator.logout('Logout', 'sidebar')
            st.divider()
        st.write('# ログインしました!')

    elif st.session_state["authentication_status"] is False:
        ## ログイン成功ログイン失敗
        st.error('Username/password is incorrect')

    elif st.session_state["authentication_status"] is None:
        ## デフォルト
        st.warning('Please enter your username and password')
except Exception as e:
    st.error(e)



with open('config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)


# OpenAI APIキーの取得
openai.api_key = api_key

if not api_key:
    st.error("OPENAI_API_KEY が設定されていません。環境変数または .env ファイルを確認してください。")
    st.stop()

# OpenAI APIキーの設定
openai.api_key = api_key

# ChatOpenAIの初期化
llm = ChatOpenAI(
    openai_api_key=api_key,
    model_name="gpt-4",
)

def initialize_vector_store() -> Chroma:
    """Initialize the VectorStore."""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store_path = "./resources/note.db"

    if Path(vector_store_path).exists():
        vector_store = Chroma(embedding_function=embeddings, persist_directory=vector_store_path)
    else:
        # print()
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

def main() -> None:

    # サイドバーメニューの作成
    menu = ["ホーム", "ヘルプ"]
    choice = st.sidebar.selectbox("メニュー", menu)

    
    if choice == "ホーム":

        """Streamlitを使用したChatGPTのメイン関数。"""
        chain = initialize_chain()



        # チャット履歴の初期化
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # ユーザー入力の監視
        user_input = st.text_input("聞きたいことを入力してください:")
        if user_input:
            # st.session_state.messages.append(HumanMessage(content=user_input))
            st.session_state.messages.append({"role": "user", "content": HumanMessage(content=user_input)})
            with st.spinner("GPTが入力中です..."):
                # response = chain(user_input)
                ai_response = chain(user_input)

            # st.session_state.messages.append(AIMessage(content=response.content))
            
            st.session_state.messages.append({"role": "assistant", "content": AIMessage(content=ai_response.content)})

        # メッセージの表示
        for msg in st.session_state.messages:
            print(msg)
            if msg['role'] == 'user':
                message(msg['content'].content, is_user=True, avatar_style="personas")
            elif msg['role'] == 'assistant':
                message(msg['content'].content, is_user=False, avatar_style="bottts")
    
    elif choice == "ヘルプ":
        st.title("ヘルプ")
        st.write("ここにヘルプ情報を記載します。")



if __name__ == "__main__":
    main()



g=embeddings, persist_directory=vector_store_path
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

def main() -> None:
    # ユーザーのログイン処理
    login_status = login_user()

    if not login_status:
        st.stop()  # ログインしないとアプリを動かさない

    # サイドバーメニューの作成
    menu = ["ホーム", "ヘルプ"]
    choice = st.sidebar.selectbox("メニュー", menu)

    if choice == "ホーム":
        """Streamlitを使用したChatGPTのメイン関数。"""
        chain = initialize_chain()

        # チャット履歴の初期化
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # ユーザー入力の監視
        user_input = st.text_input("聞きたいことを入力してください:")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": HumanMessage(content=user_input)})
            with st.spinner("GPTが入力中です..."):
                ai_response = chain(user_input)

            st.session_state.messages.append({"role": "assistant", "content": AIMessage(content=ai_response.content)})

        # メッセージの表示
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                message(msg['content'].content, is_user=True, avatar_style="personas")
            elif msg['role'] == 'assistant':
                message(msg['content'].content, is_user=False, avatar_style="bottts")

    elif choice == "ヘルプ":
        st.title("ヘルプ")
        st.write("ここにヘルプ情報を記載します。")

if __name__ == "__main__":
    main()

