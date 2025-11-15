# src/preprocessing.py
import pandas as pd
from datetime import timedelta

def load_data(uploaded_file):
    """Load CSV uploaded by Streamlit. Return raw DataFrame."""
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', low_memory=False)
    return df

def standardize_columns(df):
    """Rename common variants to a standard set we expect."""
    col_map = {}
    for c in df.columns:
        c_lower = c.strip().lower()
        if c_lower in ("customer id", "customerid", "customer_id"):
            col_map[c] = "CustomerID"
        elif c_lower in ("invoice date", "invoicedate", "invoiceDate".lower()):
            col_map[c] = "InvoiceDate"
        elif c_lower in ("invoice",):
            col_map[c] = "Invoice"
        elif c_lower in ("quantity",):
            col_map[c] = "Quantity"
        elif c_lower in ("price", "unitprice"):
            col_map[c] = "Price"
        elif c_lower in ("country",):
            col_map[c] = "Country"
    df = df.rename(columns=col_map)
    return df

def clean_transactions(df):
    """
    Clean transactions:
     - standardize column names
     - drop rows with missing CustomerID or InvoiceDate
     - convert InvoiceDate to datetime
     - remove cancelled invoices (Invoice starts with 'C') if present
     - keep positive Quantity and Price
    """
    df = standardize_columns(df)

    # required cols for transaction-level processing
    required = ["Invoice", "InvoiceDate", "Quantity", "Price", "CustomerID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # don't raise here; return df so caller can inspect columns
        raise ValueError(f"Dataset is missing required transaction columns: {missing}")

    # parse invoice date
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "CustomerID"])

    # remove cancels (Invoice code starting with 'C')
    if df["Invoice"].dtype == object:
        df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # numeric cleaning
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)

    # keep only positive quantity & price
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    # compute total price per line
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    return df

def create_rfm(df, snapshot_date=None):
    """
    Given cleaned transaction-level df, compute RFM aggregated per CustomerID.
    Returns rfm_df (CustomerID index) with columns: Recency (days), Frequency, Monetary.
    """
    # snapshot_date: date from which recency is measured (default: last InvoiceDate + 1 day)
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)

    agg = df.groupby("CustomerID").agg(
        Recency = ("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency = ("Invoice", "nunique"),
        Monetary = ("TotalPrice", "sum")
    ).reset_index()

    # some basic filtering
    agg = agg[agg["Monetary"] > 0]
    return agg
