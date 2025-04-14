from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from openai import OpenAI
from streamlit_chat import message
import json
from datetime import datetime

# APIキー設定
api_key = "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"
openai = OpenAI(api_key=api_key)

# ✅ 最新のファインチューニングジョブのモデルIDを取得
latest_job = openai.fine_tuning.jobs.list(limit=1).data[0]
fine_tuned_model_id = latest_job.fine_tuned_model
print("モデルID", fine_tuned_model_id)

# ✅ 推論に使用
response = openai.chat.completions.create(
    model=fine_tuned_model_id,
    messages=[
        {"role": "user", "content": "沖縄でプライベートなウェディングができる場所は？"}
    ]
)

print(response.choices[0].message.content)

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

# ✅ ファインチューン済みモデルに問い合わせる関数
def call_fine_tuned_model(user_input):
    try:
        response = openai.chat.completions.create(
            model=fine_tuned_model_id,
            messages=[
                {"role": "system", "content": "あなたは沖縄に詳しいアシスタントです。"},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"エラーが発生しました: {e}"

# ✅ メイン関数

def main():
    login_status = login_user()
    if not login_status:
        st.stop()

    menu = ["ホーム", "ヘルプ"]
    choice = st.sidebar.selectbox("メニュー", menu, key="menu_select")

    if choice == "ホーム":
        st.title("🤖 AI チャットボット")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        user_input = st.text_input("聞きたいことを入力してください:", key="user_input")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("GPTが入力中です..."):
                ai_response = call_fine_tuned_model(user_input)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

        for i, msg in enumerate(st.session_state.messages):
            if msg['role'] == 'user':
                message(msg['content'], is_user=True, avatar_style="personas", key=f"user_{i}")
            elif msg['role'] == 'assistant':
                message(msg['content'], is_user=False, avatar_style="bottts", key=f"assistant_{i}")

    elif choice == "ヘルプ":
        st.title("ヘルプ")
        st.write("ここにヘルプ情報を記載します。")

if __name__ == "__main__":
    main()

