import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Page Config & Style
# =========================
st.set_page_config(page_title="E-Commerce Analytics Dashboard", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
      .kpi-card {padding: 1rem; border-radius: 16px; border: 1px solid rgba(0,0,0,0.08);}
      .small {font-size: 0.9rem; opacity: 0.8;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("E-Commerce Analytics Dashboard")
st.caption("Dashboard interaktif untuk menampilkan performa penjualan, segmentasi pelanggan (RFM), dan kualitas pengiriman.")

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/main_data.csv",
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"],
    )
    return df

df = load_data()

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("Filter")

min_date = df["order_purchase_timestamp"].min().date()
max_date = df["order_purchase_timestamp"].max().date()

date_range = st.sidebar.date_input("Rentang Tanggal", (min_date, max_date))
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = date_range, date_range

state_filter = st.sidebar.multiselect(
    "State",
    sorted(df["customer_state"].dropna().unique()) if "customer_state" in df.columns else [],
)

pay_filter = st.sidebar.multiselect(
    "Payment Type",
    sorted(df["payment_type"].dropna().unique()) if "payment_type" in df.columns else [],
)

metric_mode = st.sidebar.radio("Mode Revenue", ["Produk (price)", "Produk + ongkir (price + freight)"])

# Revenue definition
if metric_mode == "Produk + ongkir (price + freight)":
    df["revenue_calc"] = df["price"].fillna(0) + df["freight_value"].fillna(0)
else:
    df["revenue_calc"] = df["price"].fillna(0)

# Apply filters
filtered = df[
    (df["order_purchase_timestamp"].dt.date >= start_date)
    & (df["order_purchase_timestamp"].dt.date <= end_date)
].copy()

if state_filter and "customer_state" in filtered.columns:
    filtered = filtered[filtered["customer_state"].isin(state_filter)]

if pay_filter and "payment_type" in filtered.columns:
    filtered = filtered[filtered["payment_type"].isin(pay_filter)]

# =========================
# Helper: KPI computation
# =========================
def compute_kpis(dfx: pd.DataFrame):
    total_orders = dfx["order_id"].nunique()
    total_revenue = dfx["revenue_calc"].sum()
    total_customers = dfx["customer_unique_id"].nunique()
    aov = total_revenue / total_orders if total_orders else 0
    return total_orders, total_revenue, total_customers, aov

# Compare period (previous range) for delta (nice upgrade)
days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
prev_end = pd.to_datetime(start_date) - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=days - 1)

prev = df[
    (df["order_purchase_timestamp"].dt.date >= prev_start.date())
    & (df["order_purchase_timestamp"].dt.date <= prev_end.date())
].copy()

# Apply same optional filters to prev
if state_filter and "customer_state" in prev.columns:
    prev = prev[prev["customer_state"].isin(state_filter)]
if pay_filter and "payment_type" in prev.columns:
    prev = prev[prev["payment_type"].isin(pay_filter)]

# Revenue calc for prev
if metric_mode == "Produk + ongkir (price + freight)":
    prev["revenue_calc"] = prev["price"].fillna(0) + prev["freight_value"].fillna(0)
else:
    prev["revenue_calc"] = prev["price"].fillna(0)

kpi_now = compute_kpis(filtered)
kpi_prev = compute_kpis(prev)

def pct_delta(now, prev):
    if prev == 0:
        return None
    return (now - prev) / prev * 100

# =========================
# KPI Row
# =========================
st.subheader("Key Metrics")
c1, c2, c3, c4 = st.columns(4)

d_orders = pct_delta(kpi_now[0], kpi_prev[0])
d_rev = pct_delta(kpi_now[1], kpi_prev[1])
d_cust = pct_delta(kpi_now[2], kpi_prev[2])
d_aov = pct_delta(kpi_now[3], kpi_prev[3])

c1.metric("Total Orders", f"{kpi_now[0]:,}", None if d_orders is None else f"{d_orders:+.1f}% vs prev")
c2.metric("Total Revenue", f"{kpi_now[1]:,.0f}", None if d_rev is None else f"{d_rev:+.1f}% vs prev")
c3.metric("Total Customers", f"{kpi_now[2]:,}", None if d_cust is None else f"{d_cust:+.1f}% vs prev")
c4.metric("Avg Order Value (AOV)", f"{kpi_now[3]:,.0f}", None if d_aov is None else f"{d_aov:+.1f}% vs prev")

with st.expander("Download data hasil filter"):
    st.download_button(
        label="Download CSV (filtered)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_data.csv",
        mime="text/csv",
    )

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Tren Bulanan", "RFM Segmentation", "Delivery vs Review"])

# -------------------------
# TAB 1: Monthly Trend
# -------------------------
with tab1:
    st.subheader("Tren Order & Revenue per Bulan")

    filtered["order_month"] = filtered["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        filtered.groupby("order_month")
        .agg(total_orders=("order_id", "nunique"), total_revenue=("revenue_calc", "sum"))
        .reset_index()
        .sort_values("order_month")
    )

    left, right = st.columns([2, 1])
    with left:
        fig = plt.figure(figsize=(12, 4))
        sns.lineplot(data=monthly, x="order_month", y="total_orders", marker="o")
        plt.title("Jumlah Order per Bulan")
        plt.xlabel("Bulan")
        plt.ylabel("Jumlah Order")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        fig = plt.figure(figsize=(12, 4))
        sns.lineplot(data=monthly, x="order_month", y="total_revenue", marker="o")
        plt.title("Total Revenue per Bulan")
        plt.xlabel("Bulan")
        plt.ylabel("Total Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    with right:
        st.markdown("**Ringkasan cepat :**")
        if len(monthly) >= 2:
            growth_orders = (monthly["total_orders"].iloc[-1] - monthly["total_orders"].iloc[0]) / max(monthly["total_orders"].iloc[0], 1) * 100
            growth_rev = (monthly["total_revenue"].iloc[-1] - monthly["total_revenue"].iloc[0]) / max(monthly["total_revenue"].iloc[0], 1) * 100
            st.write(f"- Perubahan orders (awal→akhir): **{growth_orders:+.1f}%**")
            st.write(f"- Perubahan revenue (awal→akhir): **{growth_rev:+.1f}%**")

        st.markdown("**Top 5 bulan revenue tertinggi :**")
        st.dataframe(
            monthly.sort_values("total_revenue", ascending=False).head(5),
            use_container_width=True
        )

# -------------------------
# TAB 2: RFM Segmentation
# -------------------------
with tab2:
    st.subheader("Customer Segmentation (RFM)")

    # Reference date = 1 day after last purchase in filtered range (or overall)
    reference_date = filtered["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = (
        filtered.groupby("customer_unique_id")
        .agg(
            Recency=("order_purchase_timestamp", lambda x: (reference_date - x.max()).days),
            Frequency=("order_id", "nunique"),
            Monetary=("revenue_calc", "sum"),
        )
        .reset_index()
    )

    # Handle edge case if too small data (e.g., heavy filtering)
    if len(rfm) < 10:
        st.warning("Data setelah filter terlalu sedikit untuk RFM yang stabil. Coba perluas rentang tanggal atau hilangkan filter.")
    else:
        # Scores (quantile binning)
        rfm["R_score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1])
        rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4])
        rfm["M_score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4])

        rfm["RFM_score"] = rfm["R_score"].astype(int) + rfm["F_score"].astype(int) + rfm["M_score"].astype(int)

        # Segment rule (simple & transparent)
        def segment(score):
            if score >= 10:
                return "High Value"
            elif score >= 7:
                return "Medium Value"
            return "Low Value"

        rfm["Segment"] = rfm["RFM_score"].apply(segment)

        seg = (
            rfm.groupby("Segment")
            .agg(
                total_customers=("customer_unique_id", "nunique"),
                total_revenue=("Monetary", "sum"),
                avg_revenue_per_customer=("Monetary", "mean"),
                avg_frequency=("Frequency", "mean"),
                avg_recency=("Recency", "mean"),
            )
            .reset_index()
        )

        colA, colB = st.columns([1.2, 1])
        with colA:
            fig = plt.figure(figsize=(8, 4))
            sns.barplot(data=seg, x="Segment", y="total_revenue")
            plt.title("Total Revenue per Segmen (RFM)")
            plt.xlabel("Segmen")
            plt.ylabel("Total Revenue")
            plt.tight_layout()
            st.pyplot(fig)

        with colB:
            st.markdown("**Ringkasan Segmen:**")
            st.dataframe(seg, use_container_width=True)

        st.markdown("**Top 10 pelanggan (Monetary tertinggi) dalam filter ini:**")
        st.dataframe(
            rfm.sort_values("Monetary", ascending=False).head(10)[
                ["customer_unique_id", "Recency", "Frequency", "Monetary", "RFM_score", "Segment"]
            ],
            use_container_width=True
        )

# -------------------------
# TAB 3: Delivery vs Review
# -------------------------
with tab3:
    st.subheader("Pengiriman vs Kepuasan (Review Score)")

    df_review = filtered.dropna(subset=["review_score"]).copy()
    if len(df_review) == 0:
        st.warning("Tidak ada review_score pada data setelah filter. Coba perluas rentang tanggal atau hilangkan filter.")
    else:
        # Ensure types
        df_review["review_score"] = df_review["review_score"].astype(int)

        def delay_category(days):
            if pd.isna(days):
                return "Unknown"
            if days <= 0:
                return "On Time / Early"
            if days <= 3:
                return "Slightly Late (1–3 days)"
            return "Late (>3 days)"

        df_review["delivery_category"] = df_review["delivery_delay_days"].apply(delay_category)

        summary = (
            df_review.groupby("delivery_category")
            .agg(
                avg_review=("review_score", "mean"),
                median_review=("review_score", "median"),
                total_reviews=("review_score", "count"),
            )
            .reset_index()
        )

        left, right = st.columns([1.2, 1])
        with left:
            fig = plt.figure(figsize=(8, 4))
            sns.boxplot(data=df_review, x="delivery_category", y="review_score")
            plt.title("Distribusi Review Score per Kategori Pengiriman")
            plt.xlabel("Kategori Pengiriman")
            plt.ylabel("Review Score")
            plt.tight_layout()
            st.pyplot(fig)

        with right:
            st.markdown("**Ringkasan Statistik:**")
            st.dataframe(summary, use_container_width=True)

        corr = df_review[["delivery_delay_days", "review_score"]].corr().iloc[0, 1]
        st.info(f"Korelasi (delivery_delay_days vs review_score): **{corr:.3f}** (indikatif, bukan kausalitas).")

st.markdown("---")
st.caption("Proyek Akhir Analisis Data E-Commerce Transaksi.")
