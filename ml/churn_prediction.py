import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

from config import CHURN_MODEL_PATH, PROCESSED_DATA_PATH

def train_churn_prediction(df):
    """
    Trains an XGBoost model on numerical features to predict churn.
    Uses SMOTE to balance the training dataset.
    """
    # Define features and target
    feature_cols = [
        'age', 'annual_income', 'spending_score', 'purchase_frequency',
        'last_purchase_days', 'total_purchases', 'average_order_value',
        'loyalty_years', 'customer_rating'
    ]
    target_col = 'churn_label'
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (recommended for stability and pipeline uniformity)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to handle class imbalance on the training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Original training class distribution: {np.bincount(y_train)}")
    print(f"Resampled training class distribution: {np.bincount(y_train_res)}")
    
    # Train XGBoost Classifier
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    xgb.fit(X_train_res, y_train_res)
    
    # Predictions and Evaluation
    y_pred = xgb.predict(X_test_scaled)
    y_prob = xgb.predict_proba(X_test_scaled)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Bundle the model, scaler, and metadata
    model_bundle = {
        'model': xgb,
        'scaler': scaler,
        'features': feature_cols,
        'metrics': {
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    }
    
    # Save model
    os.makedirs(os.path.dirname(CHURN_MODEL_PATH), exist_ok=True)
    joblib.dump(model_bundle, CHURN_MODEL_PATH)
    print(f"Churn prediction model saved to {CHURN_MODEL_PATH}")
    
    return model_bundle

if __name__ == "__main__":
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        train_churn_prediction(df)
    else:
        print("Processed data not found. Run preprocessing first.")
