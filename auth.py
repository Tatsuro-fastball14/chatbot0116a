import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

def load_config():
    """config.yaml を読み込む"""
    with open('config.yaml') as file:
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
        authenticator.login()
        if st.session_state["authentication_status"]:
            with st.sidebar:
                st.markdown(f'## ようこそ *{st.session_state["name"]}*')
                authenticator.logout('ログアウト', 'sidebar')
                st.divider()
            return True
        elif st.session_state["authentication_status"] is False:
            st.error('ユーザー名またはパスワードが間違っています')
        elif st.session_state["authentication_status"] is None:
            st.warning('ユーザー名とパスワードを入力してください')
    except Exception as e:
        st.error(f"ログインエラー: {e}")

    return False
