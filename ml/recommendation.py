import os
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
import joblib

from config import RECOMMENDATION_MODEL_PATH, PROCESSED_DATA_PATH

class CollaborativeRecommender:
    """
    A wrapper class for the Surprise SVD model to easily generate product recommendations.
    """
    def __init__(self, svd_model, all_categories):
        self.model = svd_model
        self.all_categories = all_categories

    def recommend(self, customer_id, top_n=5):
        """
        Predicts ratings for all categories for a customer and returns the top_n categories.
        """
        predictions = []
        for category in self.all_categories:
            pred = self.model.predict(customer_id, category)
            predictions.append((category, pred.est))
            
        # Sort categories by estimated rating in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [category for category, est in predictions[:top_n]]

def train_recommendation(df):
    """
    Trains a Collaborative Filtering model using Surprise SVD and saves the recommender bundle.
    """
    # Define columns
    user_col = 'customer_id'
    item_col = 'product_category'
    rating_col = 'customer_rating'
    
    # Extract unique categories
    all_categories = df[item_col].unique().tolist()
    
    # Initialize Surprise Reader
    reader = Reader(rating_scale=(1.0, 5.0))
    
    # Load dataset
    data = Dataset.load_from_df(df[[user_col, item_col, rating_col]], reader)
    trainset = data.build_full_trainset()
    
    # Train SVD model
    svd = SVD(random_state=42)
    svd.fit(trainset)
    print("SVD Collaborative Filtering model trained successfully.")
    
    # Create and save recommender bundle
    recommender = CollaborativeRecommender(svd, all_categories)
    
    # Test on a dummy customer
    test_id = df[user_col].iloc[0]
    test_recs = recommender.recommend(test_id)
    print(f"Sample recommendations for customer {test_id}: {test_recs}")
    
    # Save model
    os.makedirs(os.path.dirname(RECOMMENDATION_MODEL_PATH), exist_ok=True)
    joblib.dump(recommender, RECOMMENDATION_MODEL_PATH)
    print(f"Recommendation model saved to {RECOMMENDATION_MODEL_PATH}")
    
    return recommender

if __name__ == "__main__":
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        train_recommendation(df)
    else:
        print("Processed data not found. Run preprocessing first.")
