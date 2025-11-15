# 🛍️ Customer Segmentation using RFM + KMeans

This project performs **customer segmentation** using **RFM (Recency, Frequency, Monetary)** features and **KMeans clustering**.  
It includes a complete **Streamlit web app**, where you can upload a transactions CSV file and instantly view clusters, summaries, and visualizations.

---

## 🚀 Features

- Upload any **transaction-level CSV** (Online Retail format).
- Automatic column name detection (Invoice, InvoiceDate, Quantity, Price, CustomerID).
- Cleans data and removes cancelled or invalid invoices.
- Calculates **RFM features**:
  - **Recency** – Days since last purchase  
  - **Frequency** – Number of unique invoices  
  - **Monetary** – Total spending  
- Automatic **best k suggestion** using silhouette score.
- Interactive cluster visualization.
- Downloadable **clustered customers CSV**.
- Fully modular code:
  - `/src/preprocessing.py`
  - `/src/eda.py`
  - `/src/modeling.py`
  - `app.py`

---

## 📁 Project Structure

Customer_Segmentation/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│── src/
│ ├── preprocessing.py
│ ├── eda.py
│ └── modeling.py
│── data/
└── (optional sample CSV)

📥 Input Format

Your CSV should contain transaction-level data, with at least:

Required Column	Meaning
CustomerID	Unique customer
Invoice	Invoice number
InvoiceDate	Date of purchase
Quantity	Quantity ordered
Price	Unit price

The app automatically creates:

TotalPrice

RFM dataset

Cluster labels

📊 Output

You get:

Cluster visualization plot

Segment summary statistics

rfm_clustered.csv download

📦 Technologies Used

Python

Pandas

Streamlit

Scikit-Learn

Matplotlib

👨‍💻 Author

DAKSHTA WAKODE 
B.Tech AI & Data Science
Pune, Maharashtra, India