from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from streamlit_authenticator import Hasher, Authenticate
import os
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

# 環境変数の読み込み
load_dotenv()


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

def main():
    """Streamlit アプリのメイン関数"""
    # st.set_page_config(page_title="RAG ChatGPT")
    st.header("RAG ChatGPT")

    # ログイン機能
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

    # 新規登録フォーム
    st.sidebar.subheader("Register New Account")
    with st.sidebar.form("register_form"):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Register")
        if submit_button:
            hashed_password = Hasher([new_password]).hash()[0]
            USER_DATA["usernames"][new_username] = {"name": new_username, "password": hashed_password}
            st.success(f"User {new_username} registered successfully!")

if __name__ == "__main__":
    main()
