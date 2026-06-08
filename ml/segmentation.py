import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

from config import SEGMENTATION_MODEL_PATH, PROCESSED_DATA_PATH, DATA_PROCESSED_DIR

def run_elbow_method(df, features, max_k=10):
    """
    Computes inertia for different values of K and saves the elbow plot.
    """
    X = df[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    k_values = list(range(1, max_k + 1))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        
    # Plot Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, 'bo-', markersize=8)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    
    plot_path = os.path.join(DATA_PROCESSED_DIR, 'elbow_curve.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Elbow curve plot saved to {plot_path}")
    return inertias

def train_segmentation(df, features):
    """
    Trains K-Means Clustering on features, maps clusters to labels,
    saves the model, and returns customer IDs with segment labels.
    """
    X = df[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit K-Means with K=5
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    labels = kmeans.labels_
    
    # Map clusters to semantic labels: Premium, Loyal, At-Risk, New, Dormant
    # Let's compute cluster centroids in the original feature space
    centroids = []
    for i in range(5):
        cluster_data = X[labels == i]
        centroids.append({
            'cluster_id': i,
            'annual_income': cluster_data['annual_income'].mean(),
            'spending_score': cluster_data['spending_score'].mean(),
            'purchase_frequency': cluster_data['purchase_frequency'].mean()
        })
        
    # Define mapping logic:
    # 1. Premium: High annual_income AND High spending_score
    # 2. Loyal: High purchase_frequency
    # 3. Dormant: Low spending_score and Low purchase_frequency
    # 4. At-Risk: Moderate income, Low/Moderate purchase_frequency
    # 5. New: Remaining
    
    # Sort centroids to map them deterministically
    # Premium: highest income and high spending score (let's sort by income * spending_score)
    sorted_by_premium = sorted(centroids, key=lambda c: c['annual_income'] * c['spending_score'], reverse=True)
    premium_cluster = sorted_by_premium[0]['cluster_id']
    
    remaining = [c for c in centroids if c['cluster_id'] != premium_cluster]
    
    # Loyal: highest purchase frequency of the remaining
    sorted_by_loyal = sorted(remaining, key=lambda c: c['purchase_frequency'], reverse=True)
    loyal_cluster = sorted_by_loyal[0]['cluster_id']
    
    remaining = [c for c in remaining if c['cluster_id'] != loyal_cluster]
    
    # Dormant: lowest spending score of the remaining
    sorted_by_dormant = sorted(remaining, key=lambda c: c['spending_score'])
    dormant_cluster = sorted_by_dormant[0]['cluster_id']
    
    remaining = [c for c in remaining if c['cluster_id'] != dormant_cluster]
    
    # At-Risk: lower purchase frequency of the remaining
    sorted_by_at_risk = sorted(remaining, key=lambda c: c['purchase_frequency'])
    at_risk_cluster = sorted_by_at_risk[0]['cluster_id']
    
    # New: the final remaining one
    new_cluster = [c for c in remaining if c['cluster_id'] != at_risk_cluster][0]['cluster_id']
    
    # Create cluster mapping dictionary
    cluster_mapping = {
        premium_cluster: 'Premium',
        loyal_cluster: 'Loyal',
        dormant_cluster: 'Dormant',
        at_risk_cluster: 'At-Risk',
        new_cluster: 'New'
    }
    
    print("Cluster to Segment Mapping:")
    for cid, label in cluster_mapping.items():
        centroid_info = next(c for c in centroids if c['cluster_id'] == cid)
        print(f"  Cluster {cid} -> {label}: Income={centroid_info['annual_income']:.2f}, "
              f"Spending={centroid_info['spending_score']:.2f}, Freq={centroid_info['purchase_frequency']:.2f}")
              
    # Save model artifacts (scaler, kmeans model, mapping)
    model_artifact = {
        'scaler': scaler,
        'kmeans': kmeans,
        'cluster_mapping': cluster_mapping,
        'features': features
    }
    
    os.makedirs(os.path.dirname(SEGMENTATION_MODEL_PATH), exist_ok=True)
    joblib.dump(model_artifact, SEGMENTATION_MODEL_PATH)
    print(f"Segmentation model saved to {SEGMENTATION_MODEL_PATH}")
    
    # Assign labels to original dataframe
    df_result = pd.DataFrame({
        'customer_id': df['customer_id'],
        'segment': [cluster_mapping[label] for label in labels]
    })
    
    return df_result

if __name__ == "__main__":
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        features = ['annual_income', 'spending_score', 'purchase_frequency']
        run_elbow_method(df, features)
        train_segmentation(df, features)
    else:
        print("Processed data not found. Run preprocessing first.")
