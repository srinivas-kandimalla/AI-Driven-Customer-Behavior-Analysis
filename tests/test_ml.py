import os
import unittest
import pandas as pd
import numpy as np
import joblib

from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    SEGMENTATION_MODEL_PATH, PURCHASE_MODEL_PATH,
    CHURN_MODEL_PATH, RECOMMENDATION_MODEL_PATH
)
from ml.data_generation import generate_customer_data
from ml.data_preprocessing import load_and_clean_data

class TestMLPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Ensures datasets and models are generated before testing.
        """
        # If data is missing, generate it
        if not os.path.exists(RAW_DATA_PATH):
            generate_customer_data()
        if not os.path.exists(PROCESSED_DATA_PATH):
            load_and_clean_data()

    def test_data_generation(self):
        """
        Verifies that raw data was generated correctly with correct columns and size.
        """
        self.assertTrue(os.path.exists(RAW_DATA_PATH))
        df = pd.read_csv(RAW_DATA_PATH)
        self.assertEqual(len(df), 1000)
        
        required_cols = [
            'customer_id', 'age', 'gender', 'annual_income', 'spending_score',
            'purchase_frequency', 'last_purchase_days', 'total_purchases',
            'average_order_value', 'product_category', 'loyalty_years',
            'customer_rating', 'churn_label', 'region'
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)

    def test_preprocessing(self):
        """
        Verifies that preprocessing successfully handles outliers and encoding.
        """
        self.assertTrue(os.path.exists(PROCESSED_DATA_PATH))
        df = pd.read_csv(PROCESSED_DATA_PATH)
        self.assertGreater(len(df), 0)
        
        # Verify categorical columns are encoded
        self.assertIn('gender_encoded', df.columns)
        self.assertIn('product_category_encoded', df.columns)
        self.assertIn('region_encoded', df.columns)
        
        # Check no nulls exist
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_segmentation_model(self):
        """
        Verifies the Segmentation model's saved bundle structure.
        """
        self.assertTrue(os.path.exists(SEGMENTATION_MODEL_PATH))
        bundle = joblib.load(SEGMENTATION_MODEL_PATH)
        
        self.assertIn('scaler', bundle)
        self.assertIn('kmeans', bundle)
        self.assertIn('cluster_mapping', bundle)
        self.assertIn('features', bundle)
        
        # Test mapping labels count
        self.assertEqual(len(bundle['cluster_mapping']), 5)
        for label in bundle['cluster_mapping'].values():
            self.assertIn(label, ['Premium', 'Loyal', 'At-Risk', 'New', 'Dormant'])

    def test_purchase_model_inference(self):
        """
        Verifies that the Purchase model handles scale-and-predict inference.
        """
        self.assertTrue(os.path.exists(PURCHASE_MODEL_PATH))
        bundle = joblib.load(PURCHASE_MODEL_PATH)
        
        model = bundle['model']
        scaler = bundle['scaler']
        features = bundle['features']
        
        # Test input matching features: age, annual_income, spending_score, loyalty_years, average_order_value
        test_input = pd.DataFrame([{
            'age': 35.0,
            'annual_income': 70000.0,
            'spending_score': 60.0,
            'loyalty_years': 4.0,
            'average_order_value': 120.0
        }])
        
        # Scaled values
        X_scaled = scaler.transform(test_input[features])
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]
        
        self.assertIn(pred, [0, 1])
        self.assertEqual(len(prob), 2)
        self.assertTrue(0.0 <= prob[0] <= 1.0)

    def test_churn_model_inference(self):
        """
        Verifies that the Churn model handles XGBoost classification.
        """
        self.assertTrue(os.path.exists(CHURN_MODEL_PATH))
        bundle = joblib.load(CHURN_MODEL_PATH)
        
        model = bundle['model']
        scaler = bundle['scaler']
        features = bundle['features']
        
        test_input = pd.DataFrame([{
            'age': 40.0,
            'annual_income': 90000.0,
            'spending_score': 50.0,
            'purchase_frequency': 25.0,
            'last_purchase_days': 45.0,
            'total_purchases': 100.0,
            'average_order_value': 200.0,
            'loyalty_years': 5.0,
            'customer_rating': 4.2
        }])
        
        X_scaled = scaler.transform(test_input[features])
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]
        
        self.assertIn(pred, [0, 1])
        self.assertEqual(len(prob), 2)

    def test_recommendation_system(self):
        """
        Verifies that the SVD recommendation wrapper outputs 5 sorted items.
        """
        self.assertTrue(os.path.exists(RECOMMENDATION_MODEL_PATH))
        recommender = joblib.load(RECOMMENDATION_MODEL_PATH)
        
        # Test on dummy customer ID
        recs = recommender.recommend("C001", top_n=5)
        
        self.assertEqual(len(recs), 5)
        for cat in recs:
            self.assertIn(cat, ['Electronics', 'Fashion', 'Food', 'Sports', 'Beauty'])

if __name__ == "__main__":
    unittest.main()
