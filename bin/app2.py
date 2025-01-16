import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, flash, redirect, url_for
import payjp

# .envファイルを読み込む
load_dotenv()

app = Flask(__name__)

# PAY.JP APIキー設定
payjp.api_key = os.getenv("PAYJP_SECRET_KEY")  # .envから秘密鍵を取得

@app.route('/')
def index():
    return "Welcome to the Subscription Service!"

# 例: サブスクリプション処理
@app.route('/subscribe', methods=['GET', 'POST'])
def subscribe():
    if request.method == 'POST':
        payjp_token = request.form['payjp-token']
        # その他の処理
    return render_template('subscribe.html')

if __name__ == "__main__":
    app.run(debug=True)
