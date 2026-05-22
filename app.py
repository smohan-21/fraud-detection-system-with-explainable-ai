import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Fraud Detection Dashboard",
                   page_icon="🔍", layout="wide")

# load data
@st.cache_data
def load_data():
    df = pd.read_csv("sample_data.csv")
    return df

df = load_data()

# sidebar
st.sidebar.title("🔍 Fraud Detection System")
page = st.sidebar.selectbox("Navigate", 
                             ["Overview", 
                              "Transaction Explorer", 
                              "SHAP Explainer"])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
risk_filter = st.sidebar.multiselect("Risk Tier",
                                      options=["Critical Risk", 
                                               "Suspicious", "Clear"],
                                      default=["Critical Risk", 
                                               "Suspicious", "Clear"])

# filter data
filtered_df = df[df["RiskTier"].isin(risk_filter)]

# ==================
# PAGE 1 - OVERVIEW
# ==================
if page == "Overview":
    st.title("🔍 Fraud Detection System — Overview")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Transactions", 
                  f"{len(filtered_df):,}")
    with col2:
        fraud_count = filtered_df["isFraud"].sum()
        st.metric("Total Fraud Cases", 
                  f"{int(fraud_count):,}")
    with col3:
        detection_rate = (fraud_count / len(filtered_df) * 100)
        st.metric("Detection Rate", 
                  f"{detection_rate:.2f}%")
    with col4:
        avg_fraud_amt = filtered_df[filtered_df["isFraud"]==1]["TransactionAmt"].mean()
        st.metric("Avg Fraud Amount", 
                  f"${avg_fraud_amt:.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        tier_counts = filtered_df["RiskTier"].value_counts()
        fig = px.pie(values=tier_counts.values,
                     names=tier_counts.index,
                     title="Risk Tier Distribution",
                     hole=0.4,
                     color=tier_counts.index,
                     color_discrete_map={
                         "Critical Risk": "#FF4444",
                         "Suspicious": "#FFA500",
                         "Clear": "#44BB44"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(filtered_df, 
                            x="TransactionAmt",
                            color="RiskTier",
                            title="Transaction Amount Distribution",
                            log_x=True,
                            color_discrete_map={
                                "Critical Risk": "#FF4444",
                                "Suspicious": "#FFA500",
                                "Clear": "#44BB44"})
        st.plotly_chart(fig2, use_container_width=True)

    # hour of day chart
    fig3 = px.histogram(filtered_df,
                        x="HourOfDay",
                        color="RiskTier",
                        title="Fraud Pattern by Hour of Day",
                        color_discrete_map={
                            "Critical Risk": "#FF4444",
                            "Suspicious": "#FFA500",
                            "Clear": "#44BB44"})
    st.plotly_chart(fig3, use_container_width=True)

# ==============================
# PAGE 2 - TRANSACTION EXPLORER
# ==============================
elif page == "Transaction Explorer":
    st.title("🔎 Transaction Explorer")
    st.markdown("---")

    search_id = st.text_input("Search by TransactionID")

    if search_id:
        result = filtered_df[filtered_df["TransactionID"].astype(str) == search_id]
        if len(result) > 0:
            st.success(f"Transaction Found!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Transaction Amount", 
                          f"${result['TransactionAmt'].values[0]:.2f}")
            with col2:
                st.metric("Risk Tier", 
                          result["RiskTier"].values[0])
            with col3:
                st.metric("Fraud Probability", 
                          f"{result['FraudProbability'].values[0]:.4f}")
        else:
            st.error("Transaction ID not found!")

    st.subheader("Transaction Table")
    amt_range = st.slider("Filter by Transaction Amount",
                          float(filtered_df["TransactionAmt"].min()),
                          float(filtered_df["TransactionAmt"].max()),
                          (float(filtered_df["TransactionAmt"].min()),
                           float(filtered_df["TransactionAmt"].max())))

    table_df = filtered_df[
        (filtered_df["TransactionAmt"] >= amt_range[0]) &
        (filtered_df["TransactionAmt"] <= amt_range[1])
    ][["TransactionID", "TransactionAmt", 
       "RiskTier", "FraudProbability", 
       "isFraud", "HourOfDay"]].head(100)

    st.dataframe(table_df, use_container_width=True)

    fig = px.scatter(filtered_df.head(1000),
                     x="HourOfDay",
                     y="TransactionAmt",
                     color="RiskTier",
                     title="TransactionAmt vs HourOfDay",
                     color_discrete_map={
                         "Critical Risk": "#FF4444",
                         "Suspicious": "#FFA500",
                         "Clear": "#44BB44"})
    st.plotly_chart(fig, use_container_width=True)

# ==========================
# PAGE 3 - SHAP EXPLAINER
# ==========================
elif page == "SHAP Explainer":
    st.title("🧠 SHAP Explainer")
    st.markdown("---")

    trans_id = st.text_input("Enter TransactionID for SHAP Explanation")

    if trans_id:
        result = filtered_df[filtered_df["TransactionID"].astype(str) == trans_id]
        if len(result) > 0:
            prob = result["FraudProbability"].values[0]
            tier = result["RiskTier"].values[0]
            amt = result["TransactionAmt"].values[0]

            st.subheader("Transaction Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fraud Probability", f"{prob:.4f}")
            with col2:
                st.metric("Risk Tier", tier)
            with col3:
                st.metric("Amount", f"${amt:.2f}")

            st.subheader("Plain English Explanation")
            if tier == "Critical Risk":
                st.error(f"🔴 This transaction has a very high fraud probability of {prob:.2%}. "
                         f"The transaction amount of ${amt:.2f} and timing pattern "
                         f"are strong indicators of fraudulent activity. "
                         f"Immediate review is recommended.")
            elif tier == "Suspicious":
                st.warning(f"🟡 This transaction has a moderate fraud probability of {prob:.2%}. "
                           f"Some features of this transaction match known fraud patterns. "
                           f"Manual review is suggested.")
            else:
                st.success(f"🟢 This transaction appears legitimate with only {prob:.2%} "
                           f"fraud probability. No immediate action required.")
        else:
            st.error("Transaction ID not found!")

    st.subheader("Overall SHAP Feature Importance")
    top_features = pd.DataFrame({
        "Feature": ["C4", "V279", "C14", "C1", "D4",
                    "card1", "TransactionAmt", "C11",
                    "V12", "card6"],
        "Importance": [1.8, 1.5, 1.2, 1.0, 0.9,
                       0.8, 0.7, 0.6, 0.5, 0.4]
    })

    fig = px.bar(top_features,
                 x="Importance",
                 y="Feature",
                 orientation="h",
                 title="Top 10 SHAP Feature Importance",
                 color="Importance",
                 color_continuous_scale="Reds")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
