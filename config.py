import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Ensure directories exist
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# Data file paths
RAW_DATA_PATH = os.path.join(DATA_RAW_DIR, "customers.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_PROCESSED_DIR, "cleaned_data.csv")

# SQLite Database path
DATABASE_PATH = os.path.join(DATA_DIR, "customer_behavior.db")

# Model directory
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Model paths
SEGMENTATION_MODEL_PATH = os.path.join(MODEL_DIR, "segmentation_model.pkl")
PURCHASE_MODEL_PATH = os.path.join(MODEL_DIR, "purchase_model.pkl")
CHURN_MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
RECOMMENDATION_MODEL_PATH = os.path.join(MODEL_DIR, "recommendation_model.pkl")

# Scaler and Preprocessor metadata paths (for consistency in API predictions)
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

# API Settings
import os

API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("PORT", 5000))
DEBUG = False
