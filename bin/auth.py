import streamlit as st
import streamlit_authenticator as stauth

# def hash_passwords(passwords):
    """複数のパスワードを個別にハッシュ化する"""
    return [stauth.Hasher([password]).hash()[0] for password in passwords]

def init_auth():
    """認証システムを初期化し、authenticatorインスタンスを返す"""
    # ユーザー情報を設定
    names = ["Alice", "Bob"]
    usernames = ["alice123", "bob456"]
    passwords = ["password1", "password2"]  # 平文のパスワード

    # パスワードをハッシュ化
    hashed_passwords = hash_passwords(passwords)

    # 認証情報を作成
    authenticator = stauth.Authenticate(
        credentials={
            "usernames": {
                usernames[0]: {"name": names[0], "password": hashed_passwords[0]},
                usernames[1]: {"name": names[1], "password": hashed_passwords[1]},
            }
        },
        cookie_name="auth_cookie",
        cookie_expiry_days=30,
        key="secure_random_key",  # 本番環境では安全なキーを使用
    )
    return authenticator
