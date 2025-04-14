from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from openai import OpenAI
from streamlit_chat import message
from openai.types.chat import ChatCompletionMessageParam
from datetime import datetime

# ✅ ファインチューニングモデルID（あなたのモデルIDに変更）
FINE_TUNED_MODEL_ID = "ft:gpt-3.5-turbo-1106:personal::BH9JWbYY"

# ✅ OpenAIクライアント
openai_client = OpenAI(api_key="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")
                               
  # 環境変数やsecrets管理を推奨


# ✅ ログイン処理
def load_config():
    config_path = "config.yaml"
    if not Path(config_path).exists():
        st.error(f"{config_path} が見つかりません。")
        st.stop()
    with open(config_path) as file:
        return yaml.load(file, Loader=SafeLoader)

def login_user():
    config = load_config()
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    try:
        authenticator.login(key="auth_login")
        if st.session_state.get("authentication_status"):
            with st.sidebar:
                st.markdown(f'## ようこそ *{st.session_state["name"]}*')
                authenticator.logout('ログアウト', 'sidebar', key="logout_button")
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


# ✅ ファインチューニングモデルによる応答関数
def call_finetuned_model(user_input: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model=FINE_TUNED_MODEL_ID,
            messages=[
                {"role": "system", "content": "あなたは親切な沖縄観光ガイドです。"},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAIエラー: {e}")
        return "申し訳ありません、回答の生成に失敗しました。"


# ✅ メイン関数
def main() -> None:
    login_status = login_user()
    if not login_status:
        st.stop()

    menu = ["ホーム", "ヘルプ"]
    choice = st.sidebar.selectbox("メニュー", menu, key="menu_select")

    if choice == "ホーム":
        st.title("🤖 ファインチューニングAIチャットボット")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        user_input = st.text_input("質問を入力してください:", key="user_input")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("考え中..."):
                ai_reply = call_finetuned_model(user_input)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})

        for i, msg in enumerate(st.session_state.messages):
            is_user = msg["role"] == "user"
            message(msg["content"], is_user=is_user, key=f"{msg['role']}_{i}")

    elif choice == "ヘルプ":
        st.title("ヘルプ")
        st.write("ここにヘルプ情報を記載します。")

if __name__ == "__main__":
    main()

