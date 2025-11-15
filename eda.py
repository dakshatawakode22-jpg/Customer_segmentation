# src/eda.py
import pandas as pd

def create_features(rfm_df):
    """
    Input: rfm_df with columns ['CustomerID','Recency','Frequency','Monetary']
    Output: feature DataFrame (index CustomerID) used for clustering.
    """
    # ensure types
    rfm_df = rfm_df.copy()
    rfm_df["Recency"] = pd.to_numeric(rfm_df["Recency"], errors="coerce")
    rfm_df["Frequency"] = pd.to_numeric(rfm_df["Frequency"], errors="coerce")
    rfm_df["Monetary"] = pd.to_numeric(rfm_df["Monetary"], errors="coerce")

    features = rfm_df[["CustomerID", "Recency", "Frequency", "Monetary"]].set_index("CustomerID")
    return features
