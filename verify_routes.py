import sys
import os

# Ensure we can import app
sys.path.append(os.getcwd())

from app import app
# from app.routes import main # removed

def test_routes():
    # App should now work without manual registration
    client = app.test_client()

    routes = [
        '/statistics',
        '/probability-basics',
        '/probability-distributions',
        '/statistical-inference',
        '/regression',
        '/bayesian-statistics',
        '/multivariate-statistics',
        '/statistical-learning',
        '/experimental-design'
    ]

    print("Verifying routes...")
    for route in routes:
        try:
            response = client.get(route)
            if response.status_code == 200:
                print(f"✅ {route}: OK")
            else:
                print(f"❌ {route}: FAILED ({response.status_code})")
        except Exception as e:
            print(f"❌ {route}: EXCEPTION ({str(e)})")

if __name__ == "__main__":
    test_routes()
