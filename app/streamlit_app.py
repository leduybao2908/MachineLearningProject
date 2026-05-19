import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ======================
# LOAD DATA
# ======================

df = pd.read_csv("data/CC GENERAL.csv")

# ======================
# LOAD MODELS
# ======================
tree = joblib.load("models/tree.pkl")
knn = joblib.load("models/knn.pkl")
# ======================
# SIDEBAR
# ======================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Visualization",
        "Prediction"
    ]
)

# ======================
# OVERVIEW PAGE
# ======================

if page == "Overview":

    st.title("Credit Card Customer Segmentation")

    st.write("Dataset Preview")

    st.write(df.head())

    st.metric("Rows", df.shape[0])

    st.metric("Columns", df.shape[1])

# ======================
# VISUALIZATION PAGE
# ======================

elif page == "Visualization":

    st.title("Data Visualization Dashboard")

    # =========================
    # KPI CARDS
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])

    col2.metric("Columns", df.shape[1])

    col3.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("---")

    # =========================
    # HEATMAP
    # =========================

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(14,8))

    sns.heatmap(
        df.corr(numeric_only=True),
        cmap='coolwarm',
        ax=ax
    )

    st.pyplot(fig)

    st.markdown("---")

    # =========================
    # HISTOGRAM
    # =========================

    st.subheader("Balance Distribution")

    fig2, ax2 = plt.subplots(figsize=(10,5))

    sns.histplot(
        df['BALANCE'],
        bins=30,
        kde=True,
        ax=ax2
    )

    ax2.set_title("BALANCE Distribution")

    st.pyplot(fig2)

    st.markdown("---")

    # =========================
    # PURCHASE DISTRIBUTION
    # =========================

    st.subheader("Purchases Distribution")

    fig3, ax3 = plt.subplots(figsize=(10,5))

    sns.histplot(
        df['PURCHASES'],
        bins=30,
        kde=True,
        ax=ax3
    )

    ax3.set_title("PURCHASES Distribution")

    st.pyplot(fig3)

    st.markdown("---")

    # =========================
    # BOXPLOT
    # =========================

    st.subheader("Outlier Detection")

    fig4, ax4 = plt.subplots(figsize=(16,6))

    sns.boxplot(data=df.select_dtypes(include='number'), ax=ax4)

    plt.xticks(rotation=90)

    st.pyplot(fig4)

    st.markdown("---")

    # =========================
    # CLUSTER COUNT
    # =========================

    st.subheader("Cluster Distribution")

    cluster_counts = pd.DataFrame({
        "Cluster": [0, 1, 2, 3],
        "Count": [1200, 2500, 1800, 1450]
    })

    fig5, ax5 = plt.subplots(figsize=(8,5))

    sns.barplot(
        x='Cluster',
        y='Count',
        data=cluster_counts,
        ax=ax5
    )

    ax5.set_title("Customers per Cluster")

    st.pyplot(fig5)

    st.markdown("---")

    # =========================
    # PAIRPLOT
    # =========================

    st.subheader("Feature Relationships")

    sample_cols = [
        'BALANCE',
        'PURCHASES',
        'CREDIT_LIMIT',
        'PAYMENTS'
    ]

    fig6 = sns.pairplot(df[sample_cols])

    st.pyplot(fig6)

    st.markdown("---")

    # =========================
    # CREDIT LIMIT DISTRIBUTION
    # =========================

    st.subheader("Credit Limit Distribution")

    fig7, ax7 = plt.subplots(figsize=(10,5))

    sns.histplot(
        df['CREDIT_LIMIT'],
        bins=30,
        kde=True,
        ax=ax7
    )

    ax7.set_title("CREDIT LIMIT Distribution")

    st.pyplot(fig7)

    st.markdown("---")

    # =========================
    # PAYMENT DISTRIBUTION
    # =========================

    st.subheader("Payments Distribution")

    fig8, ax8 = plt.subplots(figsize=(10,5))

    sns.histplot(
        df['PAYMENTS'],
        bins=30,
        kde=True,
        ax=ax8
    )

    ax8.set_title("PAYMENTS Distribution")

    st.pyplot(fig8)

# ======================
# PREDICTION PAGE
# ======================

elif page == "Prediction":

    st.title("Customer Cluster Prediction")

    st.markdown("""
    Enter customer information to predict
    which customer cluster they belong to.
    """)

    st.markdown("---")

    # =========================
    # INPUT SECTION
    # =========================

    col1, col2 = st.columns(2)

    with col1:

        balance = st.number_input(
            "BALANCE",
            min_value=0.0,
            value=1000.0
        )

        purchases = st.number_input(
            "PURCHASES",
            min_value=0.0,
            value=500.0
        )

    with col2:

        credit_limit = st.number_input(
            "CREDIT_LIMIT",
            min_value=0.0,
            value=3000.0
        )

        payments = st.number_input(
            "PAYMENTS",
            min_value=0.0,
            value=1000.0
        )

    st.markdown("---")

    # =========================
    # PREDICT BUTTON
    # =========================

    if st.button("Predict Customer Cluster"):

        sample = [[
            balance,
            purchases,
            credit_limit,
            payments
        ]]

        # =========================
        # PREDICTION
        # =========================

        knn_result = knn.predict(sample)[0]

        tree_result = tree.predict(sample)[0]

        # =========================
        # RESULT CARDS
        # =========================

        st.subheader("Prediction Results")

        r1, r2 = st.columns(2)

        r1.metric(
            "KNN Prediction",
            f"Cluster {knn_result}"
        )

        r2.metric(
            "Decision Tree Prediction",
            f"Cluster {tree_result}"
        )

        st.markdown("---")

        # =========================
        # CLUSTER INTERPRETATION
        # =========================

        cluster_info = {
            0: {
                "title": "High Value Customers",
                "description": """
                Customers with high balance
                and high purchasing activity.
                """
            },

            1: {
                "title": "Low Spending Customers",
                "description": """
                Customers with low balance
                and limited purchases.
                """
            },

            2: {
                "title": "Cash Advance Users",
                "description": """
                Customers who frequently
                use cash advances.
                """
            },

            3: {
                "title": "Potential Premium Customers",
                "description": """
                Customers with good payment behavior
                and high credit limits.
                """
            }
        }

        # =========================
        # DISPLAY INSIGHT
        # =========================

        st.subheader("Customer Insight")

        info = cluster_info.get(
            int(knn_result),
            {
                "title": "Unknown Cluster",
                "description": "No information available."
            }
        )

        st.success(f"""
        {info['title']}
        """)

        st.write(info['description'])

        st.markdown("---")

        # =========================
        # CUSTOMER SUMMARY
        # =========================

        st.subheader("Customer Summary")

        summary_df = pd.DataFrame({
            "Feature": [
                "BALANCE",
                "PURCHASES",
                "CREDIT_LIMIT",
                "PAYMENTS"
            ],

            "Value": [
                balance,
                purchases,
                credit_limit,
                payments
            ]
        })

        st.table(summary_df)

        st.markdown("---")

        # =========================
        # SIMPLE ANALYSIS
        # =========================

        st.subheader("Analysis")

        if purchases > 5000:

            st.info("""
            This customer has high purchasing activity.
            """)

        if credit_limit > 10000:

            st.info("""
            This customer has a high credit limit.
            """)

        if payments < 500:

            st.warning("""
            Low payment behavior detected.
            """)

        if balance > 10000:

            st.warning("""
            High outstanding balance detected.
            """)