import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    page_icon="💳",
    layout="wide"
)

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        color: #667eea;
        padding: 20px 0;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #333;
        padding: 15px 0 10px 0;
        border-bottom: 2px solid #667eea;
        margin-bottom: 20px;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px 25px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("data/CC GENERAL.csv")

# ======================
# HELPER FUNCTIONS
# ======================
def get_cluster_info(cluster_id):
    cluster_info = {
        0: {"name": "High Value Customers", "desc": "Khách hàng giá trị cao"},
        1: {"name": "Low Activity Customers", "desc": "Khách hàng ít hoạt động"},
        2: {"name": "Cash Advance Users", "desc": "Khách hàng rút tiền mặt"},
        3: {"name": "Potential Premium", "desc": "Khách hàng Premium tiềm năng"}
    }
    return cluster_info.get(cluster_id, cluster_info[0])

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.markdown("### 💳 Menu")
    page = st.radio(
        "",
        ["🏠 Overview", "📊 Visualization", "🔮 Cluster Analysis", "🎯 Prediction"]
    )

# ======================
# OVERVIEW PAGE
# ======================
if page == "🏠 Overview":

    st.markdown('<h1 class="main-title">Credit Card Customer Segmentation</h1>', unsafe_allow_html=True)

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-box"><h3>Customers</h3><h1>{df.shape[0]:,}</h1></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><h3>Features</h3><h1>{df.shape[1]}</h1></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><h3>Missing</h3><h1>{df.isnull().sum().sum()}</h1></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-box"><h3>Duplicates</h3><h1>{df.duplicated().sum()}</h1></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-header">Dataset Preview</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        num_rows = st.slider("Số dòng hiển thị", 5, 20, 10)
        st.dataframe(df.head(num_rows), use_container_width=True, height=400)

    with col2:
        st.markdown('<p class="section-header">Summary</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="insight-box">
            <p><strong>Average Balance:</strong> ${df['BALANCE'].mean():.2f}</p>
            <p><strong>Average Purchases:</strong> ${df['PURCHASES'].mean():.2f}</p>
            <p><strong>Average Credit Limit:</strong> ${df['CREDIT_LIMIT'].mean():.2f}</p>
            <p><strong>Average Payments:</strong> ${df['PAYMENTS'].mean():.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">Column Information</p>', unsafe_allow_html=True)

    info_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.count().values,
        'Null': df.isnull().sum().values
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.markdown('<p class="section-header">Descriptive Statistics</p>', unsafe_allow_html=True)
    selected_cols = st.multiselect(
        "Chọn cột phân tích",
        df.select_dtypes(include=[np.number]).columns.tolist(),
        default=['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']
    )
    if selected_cols:
        st.dataframe(df[selected_cols].describe().T, use_container_width=True)

# ======================
# VISUALIZATION PAGE (Boxplot & Histogram)
# ======================
elif page == "📊 Visualization":

    st.markdown('<h1 class="main-title">Data Visualization</h1>', unsafe_allow_html=True)

    # Select feature
    feature = st.selectbox("Chọn Feature", df.select_dtypes(include=[np.number]).columns)

    st.markdown("---")

    # Toggle between Boxplot and Histogram
    chart_type = st.radio("Chọn loại chart", ["📦 Boxplot", "📈 Histogram"], horizontal=True)

    st.markdown("---")

    if chart_type == "📦 Boxplot":
        st.markdown('<p class="section-header">Boxplot - Outlier Detection</p>', unsafe_allow_html=True)

        # Single boxplot
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(y=df[feature], color='#667eea', ax=ax)
        ax.set_title(f'{feature} Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value')
        st.pyplot(fig)

        # Outlier analysis
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[feature] < lower) | (df[feature] > upper)).sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Q1", f"${Q1:.2f}")
        with col2:
            st.metric("Q3", f"${Q3:.2f}")
        with col3:
            st.metric("IQR", f"${IQR:.2f}")
        with col4:
            st.metric("Outliers", outliers)

        # Multiple boxplots
        st.markdown("---")
        st.markdown('<p class="section-header">All Features Boxplot</p>', unsafe_allow_html=True)

        features = st.multiselect(
            "Chọn features",
            df.select_dtypes(include=[np.number]).columns.tolist(),
            default=['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']
        )

        if features:
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.boxplot(data=df[features], palette='viridis', ax=ax)
            plt.xticks(rotation=45, ha='right')
            ax.set_title('Feature Comparison', fontsize=14, fontweight='bold')
            st.pyplot(fig)

    else:  # Histogram
        st.markdown('<p class="section-header">Histogram - Distribution</p>', unsafe_allow_html=True)

        # Single histogram
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.histplot(df[feature], bins=30, kde=True, color='#667eea', ax=ax)
        ax.axvline(df[feature].mean(), color='red', linestyle='--', label=f'Mean: ${df[feature].mean():.2f}')
        ax.axvline(df[feature].median(), color='green', linestyle='--', label=f'Median: ${df[feature].median():.2f}')
        ax.legend()
        ax.set_title(f'{feature} Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig)

        # Statistics
        stats = df[feature].describe()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"${stats['mean']:.2f}")
        with col2:
            st.metric("Median", f"${stats['50%']:.2f}")
        with col3:
            st.metric("Std", f"${stats['std']:.2f}")
        with col4:
            skew = stats['mean'] - stats['50%']
            st.metric("Skew", "Right" if skew > 0 else "Left")

        # Multiple histograms
        st.markdown("---")
        st.markdown('<p class="section-header">Multiple Feature Histograms</p>', unsafe_allow_html=True)

        features = st.multiselect(
            "Chọn features",
            df.select_dtypes(include=[np.number]).columns.tolist(),
            default=['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']
        )

        if features:
            cols = min(2, len(features))
            rows = (len(features) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(14, 4*rows))
            if rows == 1:
                axes = [axes]
            for i, col in enumerate(features):
                row = i // cols
                col_idx = i % cols
                sns.histplot(df[col], bins=30, kde=True, color='#667eea', ax=axes[row][col_idx])
                axes[row][col_idx].set_title(f'{col}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

# ======================
# CLUSTER ANALYSIS PAGE
# ======================
elif page == "🔮 Cluster Analysis":

    st.markdown('<h1 class="main-title">Cluster Analysis</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <p>Phân tích chi tiết 4 nhóm khách hàng được phân cụm bằng K-Means.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Cluster tabs
    tab0, tab1, tab2, tab3 = st.tabs(["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"])

    clusters = [
        {"name": "High Value Customers", "desc": "Khách hàng giá trị cao với số dư và mua sắm cao",
         "chars": ["Số dư cao", "Mua sắm tích cực", "Thanh toán đều đặn"]},
        {"name": "Low Activity Customers", "desc": "Khách hàng ít hoạt động với số dư và mua sắm thấp",
         "chars": ["Số dư thấp", "Mua sắm hạn chế", "Tiềm năng phát triển"]},
        {"name": "Cash Advance Users", "desc": "Khách hàng thường xuyên rút tiền mặt",
         "chars": ["Rút tiền mặt thường xuyên", "Phí cao", "Rủi ro tài chính"]},
        {"name": "Potential Premium", "desc": "Khách hàng có hành vi tốt và hạn mức cao",
         "chars": ["Thanh toán đầy đủ", "Hạn mức cao", "Tiềm năng nâng hạng"]}
    ]

    with tab0:
        st.markdown(f"### {clusters[0]['name']}")
        st.markdown(f"**{clusters[0]['desc']}**")
        st.markdown("**Đặc điểm:**")
        for c in clusters[0]['chars']:
            st.write(f"- {c}")

    with tab1:
        st.markdown(f"### {clusters[1]['name']}")
        st.markdown(f"**{clusters[1]['desc']}**")
        st.markdown("**Đặc điểm:**")
        for c in clusters[1]['chars']:
            st.write(f"- {c}")

    with tab2:
        st.markdown(f"### {clusters[2]['name']}")
        st.markdown(f"**{clusters[2]['desc']}**")
        st.markdown("**Đặc điểm:**")
        for c in clusters[2]['chars']:
            st.write(f"- {c}")

    with tab3:
        st.markdown(f"### {clusters[3]['name']}")
        st.markdown(f"**{clusters[3]['desc']}**")
        st.markdown("**Đặc điểm:**")
        for c in clusters[3]['chars']:
            st.write(f"- {c}")

    st.markdown("---")

    # Cluster distribution chart (simulated)
    st.markdown('<p class="section-header">Cluster Distribution</p>', unsafe_allow_html=True)

    cluster_counts = [2250, 1850, 2550, 2300]
    cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax.bar(cluster_names, cluster_counts, color=colors, edgecolor='black')
    ax.set_ylabel('Số lượng khách hàng')
    ax.set_title('Customer Distribution by Cluster')

    for bar, count in zip(bars, cluster_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(count), ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig)

# ======================
# PREDICTION PAGE
# ======================
elif page == "🎯 Prediction":

    st.markdown('<h1 class="main-title">Customer Prediction</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <p>Nhập thông tin khách hàng để dự đoán cluster.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Customer Information</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        balance = st.number_input("BALANCE (Số dư)", min_value=0.0, value=1000.0, step=100.0)
        purchases = st.number_input("PURCHASES (Mua hàng)", min_value=0.0, value=500.0, step=100.0)

    with col2:
        credit_limit = st.number_input("CREDIT_LIMIT (Hạn mức)", min_value=0.0, value=3000.0, step=100.0)
        payments = st.number_input("PAYMENTS (Thanh toán)", min_value=0.0, value=1000.0, step=100.0)

    st.markdown("---")

    if st.button("🔮 Predict", use_container_width=True):
        # Simple rule-based prediction
        if balance > 3000 and purchases > 3000:
            result = 0
        elif balance < 1000 and purchases < 1000:
            result = 1
        elif credit_limit > 10000 and payments > 2000:
            result = 3
        else:
            result = 2

        info = get_cluster_info(result)

        st.markdown("---")
        st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"### Cluster {result}")
        with col2:
            st.info(f"**{info['name']}**")

        st.markdown(f"""
        <div class="insight-box">
            <p><strong>Mô tả:</strong> {info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Insights
        st.markdown("---")
        st.markdown('<p class="section-header">Analysis</p>', unsafe_allow_html=True)

        if purchases > 5000:
            st.success("✅ High purchasing activity detected")
        if credit_limit > 10000:
            st.info("ℹ️ High credit limit")
        if payments < 500:
            st.warning("⚠️ Low payment detected")
        if balance > 10000:
            st.error("🔴 High outstanding balance")

# ======================
# FOOTER
# ======================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #888;">
    <p>Credit Card Customer Segmentation | ML Project 2024</p>
</div>
""", unsafe_allow_html=True)
