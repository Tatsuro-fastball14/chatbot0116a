import os

# 環境変数を取得
openai_api_key = os.getenv("OPENAI_API_KEY")
payjp_secret_key = os.getenv("PAYJP_SECRET_KEY")
payjp_public_key = os.getenv("PAYJP_PUBLIC_KEY")

# デバッグ用（本番環境では print しない）
if not openai_api_key:
    print("⚠ `OPENAI_API_KEY` が見つかりません")
if not payjp_secret_key:
    print("⚠ `PAYJP_SECRET_KEY` が見つかりません")
if not payjp_public_key:
    print("⚠ `PAYJP_PUBLIC_KEY` が見つかりません")
