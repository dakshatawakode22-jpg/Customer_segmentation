# app.py
import sys
import os
import io
import streamlit as st
import pandas as pd

# make src importable
sys.path.append(os.path.join(os.getcwd(), "src"))

from preprocessing import load_data, clean_transactions, create_rfm
from eda import create_features
from modeling import scale_features, find_best_k, build_model, assign_clusters, plot_clusters

st.set_page_config(page_title="Customer Segmentation (RFM)", layout="wide")
st.title("🛍️ Customer Segmentation (RFM-based KMeans)")

st.markdown("Upload your transactions CSV (Online Retail style). App will compute RFM and cluster customers.")

uploaded_file = st.file_uploader("📁 Upload transactions CSV", type=["csv"])
if uploaded_file is None:
    st.warning("⚠️ Please upload a CSV file to proceed.")
    st.stop()

# Load
try:
    raw = load_data(uploaded_file)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

st.subheader("Raw data preview")
st.write(raw.head())

# Clean transactions & create RFM
try:
    trans = clean_transactions(raw)
except Exception as e:
    st.error(f"Data cleaning failed: {e}")
    st.stop()

st.subheader("Cleaned transaction sample")
st.write(trans.head())

# create RFM
rfm = create_rfm(trans)
st.subheader("RFM (Recency, Frequency, Monetary) sample")
st.write(rfm.head())

# features
X = create_features(rfm)
st.subheader("Features used for clustering (first 10 rows)")
st.write(X.head(10))

# scaling
X_scaled, scaler = scale_features(X)

# auto-suggest best K
with st.expander("Choose number of clusters (auto-suggested)"):
    suggested_k, scores = find_best_k(X_scaled, k_min=2, k_max=8)
    st.write(f"Suggested best k (by silhouette): **{suggested_k}**")
    st.write("Silhouette scores by k:")
    st.write(pd.Series(scores).sort_index())

k = st.slider("Select number of clusters", min_value=2, max_value=10, value=int(suggested_k))

# build model
model = build_model(X_scaled, n_clusters=int(k))
rfm_labeled = assign_clusters(rfm, model, X_scaled)

st.subheader("Clustered RFM sample")
st.write(rfm_labeled.head(20))

# cluster plot
st.subheader("Cluster visualization")
st.pyplot(plot_clusters(rfm_labeled))

# cluster summary
st.subheader("Cluster summary (aggregate)")
summary = rfm_labeled.groupby("Cluster").agg({
    "Recency":"median",
    "Frequency":"median",
    "Monetary":["median","count"]
})
st.write(summary)

# allow download
csv = rfm_labeled.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download clustered customers (CSV)", data=csv, file_name="rfm_clustered.csv", mime="text/csv")




# pip install pandas numpy scikit-learn matplotlib seaborn streamlit openpyxl joblib
# pip install -r requirements.txt
# cd Customer_Segmentation
# venv\Scripts\activate
# streamlit run app.py



