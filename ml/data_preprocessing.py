import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, SCALER_PATH, MODEL_DIR

def load_and_clean_data(raw_path=RAW_DATA_PATH, save_path=PROCESSED_DATA_PATH):
    """
    Loads raw CSV, handles missing values, encodes categorical features,
    removes outliers using the IQR method, and saves the cleaned dataset.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}. Run data_generation first.")
        
    df = pd.read_csv(raw_path)
    
    # 1. Handle missing values (imputation)
    # Numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            
    # Categorical columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        if col != 'customer_id' and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            
    # 2. Encode categorical variables
    encoders = {}
    for col in ['gender', 'product_category', 'region']:
        if col in df.columns:
            le = LabelEncoder()
            # Fit and transform
            df[col + '_encoded'] = le.fit_transform(df[col])
            encoders[col] = le
            
    # Save encoders for mapping if needed
    encoders_path = os.path.join(MODEL_DIR, 'categorical_encoders.pkl')
    joblib.dump(encoders, encoders_path)
    
    # 3. Remove outliers using IQR method for numerical columns
    # We will perform IQR outlier removal on key behavior numerical columns
    # to avoid stripping away too many rows while keeping data clean.
    outlier_cols = ['annual_income', 'spending_score', 'purchase_frequency', 'average_order_value']
    initial_shape = df.shape
    
    for col in outlier_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filter DataFrame
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
    print(f"Outlier removal complete. Rows reduced from {initial_shape[0]} to {df.shape[0]}.")
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save cleaned data
    df.to_csv(save_path, index=False)
    print(f"Cleaned dataset saved at {save_path}")
    return df

def prepare_model_data(df, feature_cols, target_col, test_size=0.2, random_state=42, scaler_save_path=None):
    """
    Extracts features and target, performs feature scaling (StandardScaler),
    splits into train/test sets, and returns splits. Saves scaler if path is provided.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() == 2 else None
    )
    
    # Fit scaler on X_train and transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler if path is specified
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"Scaler saved at {scaler_save_path}")
        
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values

if __name__ == "__main__":
    load_and_clean_data()
