import os
import sqlite3
import pandas as pd
import numpy as np
import joblib

from config import (
    PROCESSED_DATA_PATH, DATABASE_PATH,
    SEGMENTATION_MODEL_PATH, PURCHASE_MODEL_PATH,
    CHURN_MODEL_PATH, RECOMMENDATION_MODEL_PATH
)
from ml.data_preprocessing import load_and_clean_data
from ml.segmentation import train_segmentation
from ml.purchase_prediction import train_purchase_prediction
from ml.churn_prediction import train_churn_prediction
from ml.recommendation import train_recommendation

def initialize_database(df_enriched):
    """
    Creates and populates the SQLite database with customer records.
    """
    print(f"\nInitializing SQLite database at {DATABASE_PATH}...")
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Drop existing table if any
    cursor.execute("DROP TABLE IF EXISTS customers")
    
    # Create Table
    cursor.execute("""
    CREATE TABLE customers (
        customer_id TEXT PRIMARY KEY,
        age INTEGER,
        gender TEXT,
        annual_income REAL,
        spending_score REAL,
        purchase_frequency INTEGER,
        last_purchase_days INTEGER,
        total_purchases INTEGER,
        average_order_value REAL,
        product_category TEXT,
        loyalty_years INTEGER,
        customer_rating REAL,
        churn_label INTEGER,
        region TEXT,
        segment TEXT,
        churn_probability REAL,
        will_purchase INTEGER
    )
    """)
    
    # Insert data
    df_enriched.to_sql('customers', conn, if_exists='append', index=False)
    
    # Verify records count
    cursor.execute("SELECT COUNT(*) FROM customers")
    count = cursor.fetchone()[0]
    print(f"Database initialized. Successfully loaded {count} customer rows into 'customers' table.")
    
    conn.commit()
    conn.close()

def main():
    print("=" * 60)
    print("STARTING MACHINE LEARNING PIPELINE TRAINING")
    print("=" * 60)
    
    # 1. Load and Clean Data
    print("\n--- Step 1: Preprocessing Data ---")
    df = load_and_clean_data()
    
    # 2. Run Segmentation (K-Means)
    print("\n--- Step 2: Customer Segmentation ---")
    segment_features = ['annual_income', 'spending_score', 'purchase_frequency']
    df_segments = train_segmentation(df, segment_features)
    
    # Merge segments into main df
    df = df.merge(df_segments, on='customer_id', how='left')
    
    # 3. Run Purchase Prediction (Random Forest)
    print("\n--- Step 3: Purchase Prediction ---")
    purchase_bundle = train_purchase_prediction(df)
    
    # Assign purchase predictions to main df
    rf_model = purchase_bundle['model']
    rf_scaler = purchase_bundle['scaler']
    rf_features = purchase_bundle['features']
    
    X_purchase = rf_scaler.transform(df[rf_features])
    df['will_purchase'] = rf_model.predict(X_purchase)
    
    # 4. Run Churn Prediction (XGBoost)
    print("\n--- Step 4: Churn Prediction ---")
    churn_bundle = train_churn_prediction(df)
    
    # Assign churn probabilities to main df
    xgb_model = churn_bundle['model']
    xgb_scaler = churn_bundle['scaler']
    xgb_features = churn_bundle['features']
    
    X_churn = xgb_scaler.transform(df[xgb_features])
    df['churn_probability'] = xgb_model.predict_proba(X_churn)[:, 1]
    
    # 5. Run Recommendations (SVD Collaborative Filtering)
    print("\n--- Step 5: Recommendation System ---")
    train_recommendation(df)
    
    # 6. Initialize and Populate Database
    print("\n--- Step 6: SQLite Database Population ---")
    # Clean df columns to match table schema exactly
    # Exclude temporary columns like categorical encoded ones
    clean_cols = [
        'customer_id', 'age', 'gender', 'annual_income', 'spending_score',
        'purchase_frequency', 'last_purchase_days', 'total_purchases',
        'average_order_value', 'product_category', 'loyalty_years',
        'customer_rating', 'churn_label', 'region', 'segment',
        'churn_probability', 'will_purchase'
    ]
    df_enriched = df[clean_cols].copy()
    initialize_database(df_enriched)
    
    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
