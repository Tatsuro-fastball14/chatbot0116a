import streamlit as st
import streamlit_authenticator as stauth
import stripe

# Stripeの設定
stripe.api_key = 'sk_test_4eC39HqLyjWDarjtT1zdp7dc'

# ユーザ情報
names = ['John Smith', 'Rebecca Briggs']
usernames = ['jsmith', 'rbriggs']
passwords = ['123', '456']

# パスワードをハッシュ化
hasher = stauth.Hasher(passwords="honeywords.hashers.HoneywordHasher")
hashed_passwords = [hasher.hash(password) for password in passwords]

# 認証情報の辞書を作成
credentials = {
    'usernames': {
        usernames[0]: {
            'name': names[0],
            'password': hashed_passwords[0],
        },
        usernames[1]: {
            'name': names[1],
            'password': hashed_passwords[1],
        }
    }
}

authenticator = stauth.Authenticate(credentials)

def main():
    """メイン関数."""
    st.title('Streamlit 課金機能デモ')

    # ログインメソッドで入力フォームを配置
    name, authentication_status, username = authenticator.login('Login', 'main')

    # 認証が成功した場合の処理
    if authentication_status:
        st.success('Welcome *%s*' % name)
        # ログアウトボタン
        if st.button('Logout'):
            authenticator.logout('Logout', 'main')
            st.info('You have been logged out.')
        # 課金ページへのリンク
        if st.button('Proceed to Checkout'):
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'jpy',
                        'product_data': {
                            'name': 'Premium Plan',
                        },
                        'unit_amount': 2000,
                    },
                    'quantity': 1,
                }],
                mode='payment',
                success_url='https://your-domain.com/success',
                cancel_url='https://your-domain.com/cancel',
            )
            st.write('Please proceed to payment.')
            st.markdown(f"[Pay Here]({session.url})")

    # 認証が失敗した場合の処理
    elif authentication_status == False:
        st.error('Username/password is incorrect')

    # 未ログインの場合
    elif authentication_status is None:
        st.warning('Please enter your username and password')

if __name__ == "__main__":
    main()
