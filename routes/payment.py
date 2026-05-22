import os
import stripe
from dotenv import load_dotenv


# Load secret API keys from your .env file
load_dotenv()
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# Replace this with your actual local or production URL where the app runs
BASE_URL = os.getenv("APP_URL", "http://localhost:8501")

def create_checkout_session(price_id: str, user_email: str = None) -> str:
    """
    Creates a Stripe checkout session for a subscription or payment.
    Returns the secure URL to redirect the user to.
    """
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price": price_id,  # Looks like 'price_1Qx...' from Stripe Dashboard
                    "quantity": 1,
                },
            ],
            mode="subscription",  # Change to "payment" if it's a one-time fee
            customer_email=user_email,
            # App pages Stripe will bounce the user back to
            success_url=f"{BASE_URL}/?payment=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{BASE_URL}/?payment=cancelled",
        )
        return session.url
    except Exception as e:
        print(f"Error creating Stripe checkout session: {e}")
        return None

def verify_payment(session_id: str) -> bool:
    """
    Checks if the user actually completed the payment.
    Call this when the user returns via the success_url.
    """
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        return session.payment_status == "paid"
    except Exception as e:
        print(f"Error verifying payment: {e}")
        return False

# --- Quick Test Rig ---
if __name__ == "__main__":
    # Ensure you have STRIPE_SECRET_KEY in your env before running this test
    print("Testing checkout generation...")
    # Using a dummy price ID just to show syntax; replace with your real stripe price ID
    test_url = create_checkout_session(price_id="price_H5ggL2v1j6N9xZ", user_email="test@user.com")
    if test_url:
        print(f"Success! Direct your user here: {test_url}")
