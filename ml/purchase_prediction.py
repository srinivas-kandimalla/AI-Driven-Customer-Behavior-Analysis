import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from config import PURCHASE_MODEL_PATH, PROCESSED_DATA_PATH

def train_purchase_prediction(df):
    """
    Derives target 'will_purchase', scales features, splits data,
    tunes a Random Forest Classifier via GridSearchCV, prints metrics,
    and saves the model bundle.
    """
    # Define features and derive target
    feature_cols = ['age', 'annual_income', 'spending_score', 'loyalty_years', 'average_order_value']
    
    # Target is will_purchase (1 if purchase_frequency > median, else 0)
    median_freq = df['purchase_frequency'].median()
    df['will_purchase'] = (df['purchase_frequency'] > median_freq).astype(int)
    
    X = df[feature_cols].copy()
    y = df['will_purchase'].copy()
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'random_state': [42]
    }
    
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    
    # Evaluate model
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nPurchase Prediction Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create model bundle to save
    model_bundle = {
        'model': best_model,
        'scaler': scaler,
        'features': feature_cols,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }
    
    # Save model bundle
    os.makedirs(os.path.dirname(PURCHASE_MODEL_PATH), exist_ok=True)
    joblib.dump(model_bundle, PURCHASE_MODEL_PATH)
    print(f"Purchase prediction model saved to {PURCHASE_MODEL_PATH}")
    
    return model_bundle

if __name__ == "__main__":
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        train_purchase_prediction(df)
    else:
        print("Processed data not found. Run preprocessing first.")
