# src/modeling.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def scale_features(X):
    """X is DataFrame or 2D array"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def find_best_k(X_scaled, k_min=2, k_max=8):
    """Return (best_k, scores_dict) using silhouette score."""
    best_k = k_min
    best_score = -1
    scores = {}
    for k in range(k_min, min(k_max, X_scaled.shape[0] - 1) + 1):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        if len(np.unique(labels)) == 1:
            scores[k] = -1
            continue
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, scores

def build_model(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(X_scaled)
    return km

def assign_clusters(rfm_df, model, X_scaled):
    labels = model.predict(X_scaled)
    out = rfm_df.copy()
    out["Cluster"] = labels
    return out

def plot_clusters(rfm_with_clusters):
    """Return matplotlib plt object: Frequency vs Monetary colored by cluster."""
    plt.figure(figsize=(8,5))
    x = rfm_with_clusters["Frequency"]
    y = rfm_with_clusters["Monetary"]
    c = rfm_with_clusters["Cluster"]
    scatter = plt.scatter(x, y, c=c, cmap="tab10", s=35, alpha=0.8)
    plt.xlabel("Frequency (unique invoices)")
    plt.ylabel("Monetary (total spend)")
    plt.title("Customer Segments (Frequency vs Monetary)")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    return plt




# cd Customer_Segmentation
# venv\Scripts\activate
# pip install pandas numpy scikit-learn matplotlib seaborn streamlit openpyxl

