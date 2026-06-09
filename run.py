import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    API_HOST,
    API_PORT,
    DEBUG,
    SEGMENTATION_MODEL_PATH,
    PURCHASE_MODEL_PATH,
    CHURN_MODEL_PATH,
    RECOMMENDATION_MODEL_PATH,
)


def run_app():
    """
    Main entry point for starting the application.
    Checks if model pickle files are trained and saved. If not, runs training.
    Starts the Flask app.
    """
    # Check if models exist. If not, trigger trainer
    models_missing = (
        not os.path.exists(SEGMENTATION_MODEL_PATH)
        or not os.path.exists(PURCHASE_MODEL_PATH)
        or not os.path.exists(CHURN_MODEL_PATH)
        or not os.path.exists(RECOMMENDATION_MODEL_PATH)
    )

    if models_missing:
        print("Model binary files are missing. Running ML pipeline training first...")
        from ml.data_generation import generate_customer_data
        from ml.model_trainer import main as train_models

        # Ensure raw data is generated
        generate_customer_data()

        # Train all models and populate DB
        train_models()
        print("Training pipeline finished. Launching Flask Web Server...\n")

    # Start server
    # Always initialize database
    from ml.model_trainer import initialize_database

    try:
        initialize_database()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"DB init warning: {e}")

    # Start server
    from api.app import create_app

    app = create_app()

    print("=" * 60)
    print(
        f"Customer Behavior Analysis Platform starting at: http://{API_HOST}:{API_PORT}"
    )
    print(f"Open http://{API_HOST}:{API_PORT} in your browser to view the dashboard.")
    print("=" * 60)

    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)


if __name__ == "__main__":
    run_app()
