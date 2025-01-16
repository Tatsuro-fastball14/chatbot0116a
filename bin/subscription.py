import payjp
import os

payjp.api_key = os.getenv('PAYJP_SECRET_KEY')  # 環境変数からPAY.JPのAPIキーを取得

def create_customer(email, payment_method_id):
    """顧客をPAY.JPに登録する"""
    customer = payjp.Customer.create(
        email=email,
        payment_method=payment_method_id,
        invoice_settings={
            'default_payment_method': payment_method_id
        }
    )
    return customer.id

def create_plan(amount, interval):
    """サブスクリプションプランを作成する"""
    plan = payjp.Plan.create(
        amount=amount,
        currency='jpy',
        interval=interval
    )
    return plan.id

def create_subscription(customer_id, plan_id):
    """顧客に対してサブスクリプションを開始する"""
    subscription = payjp.Subscription.create(
        customer=customer_id,
        items=[{'plan': plan_id}],
    )
    return subscription

def cancel_subscription(subscription_id):
    """サブスクリプションをキャンセルする"""
    subscription = payjp.Subscription.retrieve(subscription_id)
    subscription.cancel()
    return subscription

# 実際の使用時には以下のように各種関数を呼び出すことができます。
# customer_id = create_customer('example@email.com', 'pm_card_visa')
# plan_id = create_plan(5000, 'month')  # 月額5000円のプランを作成
# subscription = create_subscription(customer_id, plan_id)ああああ

ああああ