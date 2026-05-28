import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ======================
# LOAD MODELS
# ======================
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
scaler = None
kmeans = None
knn_model = None
tree_model = None
rf_model = None

try:
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(models_dir, "kmeans.pkl"))
    knn_model = joblib.load(os.path.join(models_dir, "knn.pkl"))
    
    tree_model = joblib.load(os.path.join(models_dir, "tree.pkl"))
    if not hasattr(tree_model, "monotonic_cst"):
        tree_model.monotonic_cst = None
        
    rf_model = joblib.load(os.path.join(models_dir, "rf.pkl"))
    if not hasattr(rf_model, "monotonic_cst"):
        rf_model.monotonic_cst = None
    if hasattr(rf_model, "estimators_"):
        for estimator in rf_model.estimators_:
            if not hasattr(estimator, "monotonic_cst"):
                estimator.monotonic_cst = None
        
    models_loaded = True
except:
    models_loaded = False

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Customer Segmentation | Neural Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================
# CUSTOM CSS - Digital Banking Noir Theme
# ======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a24;
        --bg-card-hover: #22222e;
        --accent-cyan: #00f0ff;
        --accent-magenta: #ff00aa;
        --accent-gold: #ffd700;
        --accent-green: #00ff88;
        --text-primary: #ffffff;
        --text-secondary: #8a8a9a;
        --text-muted: #5a5a6a;
        --border: #2a2a3a;
        --glow-cyan: 0 0 20px rgba(0, 240, 255, 0.3);
        --glow-magenta: 0 0 20px rgba(255, 0, 170, 0.3);
    }

    * {
        font-family: 'Space Grotesk', -apple-system, sans-serif;
    }

    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
    }

    /* Main Title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-magenta));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 30px 0;
        letter-spacing: -0.02em;
        text-shadow: var(--glow-cyan);
    }

    /* Section Headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--accent-cyan);
        padding: 15px 0 10px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Metric Cards */
    .metric-box {
        background: var(--bg-card);
        border: 1px solid var(--border);
        padding: 20px 25px;
        border-radius: 12px;
        color: var(--text-primary);
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .metric-box:hover {
        border-color: var(--accent-cyan);
        box-shadow: var(--glow-cyan);
        transform: translateY(-2px);
    }
    .metric-box h3 {
        color: var(--text-secondary);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0;
    }
    .metric-box h1 {
        color: var(--accent-cyan);
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0 0 0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Insight Boxes */
    .insight-box {
        background: var(--bg-card);
        padding: 20px;
        border-radius: 12px;
        border-left: 3px solid var(--accent-magenta);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    /* Sidebar */
    .stSidebar {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }

    /* Radio Button Styling */
    .stRadio > div {
        background: var(--bg-card);
        padding: 10px;
        border-radius: 10px;
        border: 1px solid var(--border);
    }
    .stRadio label {
        color: var(--text-secondary);
        font-weight: 500;
    }
    .stRadio label:hover {
        color: var(--accent-cyan);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 5px;
        gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-cyan) !important;
        color: var(--bg-primary) !important;
        border-radius: 8px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-magenta));
        border: none;
        color: white;
        font-weight: 600;
        padding: 12px 30px;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: var(--glow-cyan), var(--glow-magenta);
        transform: scale(1.02);
    }

    /* Number Inputs */
    .stNumberInput > div > div > input {
        background: var(--bg-card);
        border: 1px solid var(--border);
        color: var(--text-primary);
        border-radius: 8px;
    }
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-cyan);
        box-shadow: var(--glow-cyan);
    }

    /* Dataframes */
    .stDataFrame {
        background: var(--bg-card);
        border-radius: 12px;
        border: 1px solid var(--border);
    }

    /* Success/Error/Warning boxes */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 10px;
    }

    /* Footer */
    footer {
        visibility: hidden;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-cyan);
    }

    /* Grid Lines */
    .element-container {
        margin-bottom: 1rem;
    }

    /* Metric Labels */
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-size: 0.85rem;
    }
    [data-testid="stMetricValue"] {
        color: var(--accent-cyan);
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD DATA
# ======================
import os
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "CC GENERAL.csv")
df = pd.read_csv(data_path)

# Drop CUST_ID as it's not numeric
if 'CUST_ID' in df.columns:
    df = df.drop('CUST_ID', axis=1)

# ======================
# HELPER FUNCTIONS
# ======================
def get_cluster_info(cluster_id):
    cluster_info = {
        0: {"name": "Low Activity Customers", "desc": "Khách hàng ít hoạt động"},
        1: {"name": "High Value Customers", "desc": "Khách hàng giá trị cao"},
        2: {"name": "Cash Advance Users", "desc": "Khách hàng rút tiền mặt"},
        3: {"name": "Potential Premium", "desc": "Khách hàng Premium tiềm năng"}
    }
    return cluster_info.get(cluster_id, cluster_info[0])

# ======================
# CLUSTER CONFIGURATION
# ======================
# Cluster-specific colors and names
CLUSTER_CONFIG = {
    0: {"name": "Low Activity Customers", "name_vn": "Khách Hàng Ít Hoạt Động", "color": "#ff00aa", "glow": "0 0 20px rgba(255, 0, 170, 0.3)"},
    1: {"name": "High Value Customers", "name_vn": "Khách Hàng Giá Trị Cao", "color": "#00f0ff", "glow": "0 0 20px rgba(0, 240, 255, 0.3)"},
    2: {"name": "Cash Advance Users", "name_vn": "Khách Hàng Rút Tiền Mặt", "color": "#ffd700", "glow": "0 0 20px rgba(255, 215, 0, 0.3)"},
    3: {"name": "Potential Premium", "name_vn": "Khách Hàng Premium Tiềm Năng", "color": "#00ff88", "glow": "0 0 20px rgba(0, 255, 136, 0.3)"}
}

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="background: linear-gradient(135deg, #00f0ff, #ff00aa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
            ◈ SEGMENT
        </h2>
        <p style="color: var(--text-secondary); font-size: 0.8rem; margin-top: 5px;">
            Customer Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "",
        [
            "◈ Overview",
            "◈ Visualization",
            "◈ Cluster Analysis",
            "◈ Prediction"
        ],
        label_visibility="hidden"
    )

    st.markdown("---")

    # Model status indicator
    if models_loaded:
        st.success("✓ Models Loaded")
    else:
        st.warning("○ Models Pending")

# ======================
# OVERVIEW PAGE
# ======================
if page == "◈ Overview":

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
elif page == "◈ Visualization":

    st.markdown('<h1 class="main-title">Data Visualization</h1>', unsafe_allow_html=True)

    # Select feature
    feature = st.selectbox("Chọn Feature", df.select_dtypes(include=[np.number]).columns)

    st.markdown("---")

    # Toggle between Boxplot and Histogram
    chart_type = st.radio("Chọn loại chart", [" Boxplot", " Histogram"], horizontal=True)

    st.markdown("---")

    if chart_type == " Boxplot":
        st.markdown('<p class="section-header">Boxplot - Outlier Detection</p>', unsafe_allow_html=True)

        # Single boxplot
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(y=df[feature], color='#00f0ff', ax=ax)
        ax.set_title(f'{feature} Distribution', fontsize=14, fontweight='bold', color='white')
        ax.set_ylabel('Value', color='white')
        ax.set_xlabel('')
        ax.tick_params(colors='white')
        ax.set_facecolor('#12121a')
        fig.patch.set_facecolor('#0a0a0f')
        for spine in ax.spines.values():
            spine.set_color('#2a2a3a')
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
        sns.histplot(df[feature], bins=30, kde=True, color='#00f0ff', ax=ax, alpha=0.7)
        ax.axvline(df[feature].mean(), color='#ff00aa', linestyle='--', linewidth=2, label=f'Mean: ${df[feature].mean():.2f}')
        ax.axvline(df[feature].median(), color='#ffd700', linestyle='--', linewidth=2, label=f'Median: ${df[feature].median():.2f}')
        ax.legend()
        ax.set_title(f'{feature} Distribution', fontsize=14, fontweight='bold', color='white')
        ax.set_ylabel('Count', color='white')
        ax.set_xlabel('Value', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#12121a')
        fig.patch.set_facecolor('#0a0a0f')
        for spine in ax.spines.values():
            spine.set_color('#2a2a3a')
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
elif page == "◈ Cluster Analysis":

    st.markdown('<h1 class="main-title">Cluster Analysis</h1>', unsafe_allow_html=True)

    feature_cols = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']

    if not models_loaded:
        st.warning("Models chua duoc load. Hien tai hien thi phan bo mau.")

        st.markdown("---")
        st.markdown('<p class="section-header">Cluster Distribution (Sample)</p>', unsafe_allow_html=True)

        np.random.seed(42)
        sample_clusters = np.random.randint(0, 4, size=len(df))
        cluster_counts = pd.Series(sample_clusters).value_counts().sort_index().values
        cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ['#00f0ff', '#ff00aa', '#ffd700', '#00ff88']

        axes[0].pie(cluster_counts, labels=cluster_names, autopct='%1.1f%%',
                   colors=colors, explode=[0.02]*4, shadow=True, startangle=90)
        axes[0].set_title('Cluster Distribution (Pie)')

        bars = axes[1].bar(cluster_names, cluster_counts, color=colors, edgecolor='black')
        axes[1].set_ylabel('So luong khach hang')
        for bar, count in zip(bars, cluster_counts):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    str(count), ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    else:
        # Preprocessing: Same as train_models.py
        feature_cols = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']

        # Create df_log for cluster prediction
        df_for_cluster = df.copy()

        # Log transform only numeric columns (same as notebook/train_models.py)
        df_for_cluster[feature_cols] = np.log1p(df_for_cluster[feature_cols])

        # Fill missing values for numeric columns only
        for feat in feature_cols:
            df_for_cluster[feat] = df_for_cluster[feat].fillna(df_for_cluster[feat].mean())

        # Use the SAVED scaler (not a new one!)
        X = df_for_cluster[feature_cols]
        X_scaled = scaler.transform(X)

        # Predict clusters using trained kmeans
        cluster_labels = kmeans.predict(X_scaled)

        # Assign cluster labels to dataframe
        df['Cluster'] = cluster_labels
        feature_cols_display = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']

        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)
        cluster_counts = df['Cluster'].value_counts().sort_index()
        total = len(df)

        for col, cid in zip([col1, col2, col3, col4], range(4)):
            cnt = cluster_counts.get(cid, 0)
            pct = cnt / total * 100
            with col:
                st.metric(f"Cluster {cid}", f"{cnt}", f"{pct:.1f}%")

        # Statistics table
        st.markdown("---")
        st.markdown('<p class="section-header">Cluster Statistics</p>', unsafe_allow_html=True)

        cluster_summary = df.groupby('Cluster')[feature_cols].agg(['mean', 'median']).round(2)
        cluster_summary.columns = ['_'.join(col) for col in cluster_summary.columns]
        st.dataframe(cluster_summary, use_container_width=True)

        # Charts
        st.markdown("---")
        chart_type = st.selectbox("Chon loai bieu do", ["Pie Chart", "Bar Chart", "Box Plot", "Heatmap", "3D Cluster Plot"])

        colors = ['#00f0ff', '#ff00aa', '#ffd700', '#00ff88']

        if chart_type == "3D Cluster Plot":
            import plotly.express as px
            plot_df = df.copy()
            cluster_names_map = {
                0: "Cluster 0: Low Activity",
                1: "Cluster 1: High Value",
                2: "Cluster 2: Cash Advance",
                3: "Cluster 3: Potential Premium"
            }
            plot_df['Customer Segment'] = plot_df['Cluster'].map(cluster_names_map)
            
            fig_3d = px.scatter_3d(
                plot_df,
                x='BALANCE',
                y='PURCHASES',
                z='CREDIT_LIMIT',
                color='Customer Segment',
                color_discrete_map={
                    "Cluster 0: Low Activity": '#ff00aa',
                    "Cluster 1: High Value": '#00f0ff',
                    "Cluster 2: Cash Advance": '#ffd700',
                    "Cluster 3: Potential Premium": '#00ff88'
                },
                hover_data=['PAYMENTS', 'TENURE'],
                labels={
                    'BALANCE': 'Balance ($)',
                    'PURCHASES': 'Purchases ($)',
                    'CREDIT_LIMIT': 'Credit Limit ($)'
                },
                height=700
            )
            fig_3d.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Space Grotesk, sans-serif", color='#ffffff'),
                scene=dict(
                    xaxis=dict(backgroundcolor="#12121a", gridcolor="#2a2a3a", showbackground=True),
                    yaxis=dict(backgroundcolor="#12121a", gridcolor="#2a2a3a", showbackground=True),
                    zaxis=dict(backgroundcolor="#12121a", gridcolor="#2a2a3a", showbackground=True),
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(10,10,15,0.8)"
                )
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            if chart_type == "Pie Chart":
                axes[0].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index],
                           autopct='%1.1f%%', colors=colors[:len(cluster_counts)],
                           explode=[0.02]*len(cluster_counts), shadow=True, startangle=90)
                axes[0].set_title('Cluster Distribution')
            elif chart_type == "Bar Chart":
                bars = axes[0].bar([f'Cluster {i}' for i in cluster_counts.index], cluster_counts.values,
                                  color=colors[:len(cluster_counts)], edgecolor='black')
                axes[0].set_ylabel('So luong khach hang')
                for bar, cnt in zip(bars, cluster_counts.values):
                    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                            str(cnt), ha='center', fontweight='bold')
            elif chart_type == "Box Plot":
                df_melted = df.melt(id_vars=['Cluster'], value_vars=feature_cols,
                                   var_name='Feature', value_name='Value')
                sns.boxplot(data=df_melted, x='Feature', y='Value', hue='Cluster',
                           palette=colors, ax=axes[0])
                axes[0].legend(title='Cluster')
            else:
                cluster_means = df.groupby('Cluster')[feature_cols].mean()
                cluster_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
                sns.heatmap(cluster_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                           ax=axes[0], cbar_kws={'label': 'Normalized'})

            # Feature comparison
            for feat in feature_cols:
                means = df.groupby('Cluster')[feat].mean()
                axes[1].plot(means.index, means.values, 'o-', label=feat, linewidth=2, markersize=8)
            axes[1].set_xlabel('Cluster')
            axes[1].set_ylabel('Gia tri trung binh')
            axes[1].legend(loc='best')
            axes[1].set_xticks(range(4))
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

        # Cluster details tabs
        st.markdown("---")
        st.markdown('<p class="section-header">Cluster Details</p>', unsafe_allow_html=True)

        # Custom styled tabs
        tab_labels = ["◈ Cluster 0", "◈ Cluster 1", "◈ Cluster 2", "◈ Cluster 3"]
        tabs = st.tabs(tab_labels)

        for tab, cid in zip(tabs, [0, 1, 2, 3]):
            config = CLUSTER_CONFIG[cid]
            with tab:
                cluster_data = df[df['Cluster'] == cid]

                # Cluster header card
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {config['color']}15, {config['color']}05);
                           padding: 25px; border-radius: 15px; border: 1px solid {config['color']}50;
                           margin-bottom: 20px;">
                    <h2 style="color: {config['color']}; margin: 0 0 10px 0; font-size: 1.5rem;">
                        {config['name']}
                    </h2>
                    <p style="color: #8a8a9a; margin: 0;">
                        {config['name_vn']} | {len(cluster_data)} customers ({len(cluster_data)/total*100:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    for feat in feature_cols:
                        st.metric(f"Avg {feat}", f"${cluster_data[feat].mean():,.2f}")
                with col2:
                    for feat in feature_cols:
                        st.metric(f"Med {feat}", f"${cluster_data[feat].median():,.2f}")

                # Bar chart cho cluster
                fig, ax = plt.subplots(figsize=(8, 4))
                means = cluster_data[feature_cols].mean()
                ax.barh(feature_cols, means.values, color=config['color'], edgecolor='white', alpha=0.8)
                ax.set_xlabel('Gia tri trung binh', color='white')
                ax.set_title(f'Cluster {cid} - Feature Means', color='white')
                ax.tick_params(colors='white')
                ax.set_facecolor('#12121a')
                fig.patch.set_facecolor('#0a0a0f')
                for spine in ax.spines.values():
                    spine.set_color('#2a2a3a')
                for i, v in enumerate(means.values):
                    ax.text(v + 50, i, f'${v:,.0f}', va='center', color='white')
                st.pyplot(fig)

# ======================
# PREDICTION PAGE
# ======================
elif page == "◈ Prediction":

    st.markdown('<h1 class="main-title">Customer Prediction</h1>', unsafe_allow_html=True)

    if not models_loaded:
        st.warning("⚠️ Models chưa được load. Vui lòng chạy notebook để train và lưu models trước.")
    else:
        st.markdown("""
        <div class="insight-box">
            <p>Nhập thông tin khách hàng để dự đoán cluster bằng 3 mô hình: KNN, Decision Tree, Random Forest.</p>
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

        if st.button("Predict", use_container_width=True):
            # Create input array
            new_customer = np.array([[balance, purchases, credit_limit, payments]])

            # Apply log transform (same as training)
            new_customer_log = np.log1p(new_customer)

            # Prepare input data with scaling
            new_customer_scaled = scaler.transform(new_customer_log)

            # Predict with all models
            knn_result = int(knn_model.predict(new_customer_scaled)[0])
            tree_result = int(tree_model.predict(new_customer_scaled)[0])
            rf_result = int(rf_model.predict(new_customer_scaled)[0])

            st.markdown("---")
            st.markdown('<p class="section-header">Prediction Results</p>', unsafe_allow_html=True)

            # Display results in columns
            col1, col2, col3 = st.columns(3)

            knn_info = get_cluster_info(knn_result)
            tree_info = get_cluster_info(tree_result)
            rf_info = get_cluster_info(rf_result)

            with col1:
                cluster_config = CLUSTER_CONFIG[knn_result]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {cluster_config['color']}20, {cluster_config['color']}05);
                           padding:25px; border-radius:15px; text-align:center; border: 2px solid {cluster_config['color']}60;
                           box-shadow: {cluster_config['glow']};">
                    <h3 style="color: #00f0ff; margin: 0 0 10px 0; font-size: 0.9rem;">KNN</h3>
                    <div style="font-size: 4rem; font-weight: 700; color: {cluster_config['color']};">{knn_result}</div>
                    <div style="height: 3px; background: {cluster_config['color']}; margin: 10px 0; border-radius: 2px;"></div>
                    <p style="color: {cluster_config['color']}; margin: 0; font-weight: 600; font-size: 0.85rem;">{cluster_config['name']}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                cluster_config = CLUSTER_CONFIG[tree_result]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {cluster_config['color']}20, {cluster_config['color']}05);
                           padding:25px; border-radius:15px; text-align:center; border: 2px solid {cluster_config['color']}60;
                           box-shadow: {cluster_config['glow']};">
                    <h3 style="color: #00ff88; margin: 0 0 10px 0; font-size: 0.9rem;">Decision Tree</h3>
                    <div style="font-size: 4rem; font-weight: 700; color: {cluster_config['color']};">{tree_result}</div>
                    <div style="height: 3px; background: {cluster_config['color']}; margin: 10px 0; border-radius: 2px;"></div>
                    <p style="color: {cluster_config['color']}; margin: 0; font-weight: 600; font-size: 0.85rem;">{cluster_config['name']}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                cluster_config = CLUSTER_CONFIG[rf_result]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {cluster_config['color']}20, {cluster_config['color']}05);
                           padding:25px; border-radius:15px; text-align:center; border: 2px solid {cluster_config['color']}60;
                           box-shadow: {cluster_config['glow']};">
                    <h3 style="color: #ff00aa; margin: 0 0 10px 0; font-size: 0.9rem;">Random Forest</h3>
                    <div style="font-size: 4rem; font-weight: 700; color: {cluster_config['color']};">{rf_result}</div>
                    <div style="height: 3px; background: {cluster_config['color']}; margin: 10px 0; border-radius: 2px;"></div>
                    <p style="color: {cluster_config['color']}; margin: 0; font-weight: 600; font-size: 0.85rem;">{cluster_config['name']}</p>
                </div>
                """, unsafe_allow_html=True)

            # Final prediction summary
            results = [knn_result, tree_result, rf_result]
            if len(set(results)) == 1:
                final_cluster = knn_result
                cluster_config = CLUSTER_CONFIG[final_cluster]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {cluster_config['color']}30, {cluster_config['color']}10);
                           padding: 30px; border-radius: 20px; text-align: center;
                           border: 2px solid {cluster_config['color']}; margin: 20px 0;">
                    <h2 style="color: {cluster_config['color']}; margin: 0 0 15px 0; font-size: 1.8rem;">
                        Final Prediction
                    </h2>
                    <div style="font-size: 1.2rem; color: white; margin-bottom: 15px;">
                        Customer belongs to <strong style="color: {cluster_config['color']};">{cluster_config['name']}</strong>
                    </div>
                    <div style="font-size: 0.95rem; color: #8a8a9a; line-height: 1.6;">
                        {cluster_config['name_vn']}
                    </div>
                    <div style="height: 3px; background: linear-gradient(90deg, transparent, {cluster_config['color']}, transparent); margin: 20px 0;"></div>
                    <p style="color: #8a8a9a; margin: 0; font-size: 0.85rem;">
                        ✓ All 3 models agree
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #ff444430; padding: 25px; border-radius: 15px;
                           border: 1px solid #ff4444; margin: 20px 0;">
                    <h3 style="color: #ff4444; margin: 0 0 15px 0;">⚠️ Model Disagreement</h3>
                    <p style="color: #8a8a9a; margin: 0 0 15px 0;">Different models predicted different clusters:</p>
                """, unsafe_allow_html=True)

            # Insights
            st.markdown("---")
            st.markdown('<p class="section-header">Customer Analysis</p>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if purchases > 5000:
                    st.markdown("""
                    <div style="background: #00ff8820; padding: 15px 20px; border-radius: 10px; border-left: 4px solid #00ff88;">
                        <strong style="color: #00ff88;">High Purchasing Activity</strong>
                        <p style="color: #8a8a9a; margin: 5px 0 0 0;">Purchases exceed $5,000 threshold</p>
                    </div>
                    """, unsafe_allow_html=True)
                if credit_limit > 10000:
                    st.markdown("""
                    <div style="background: #ffd70020; padding: 15px 20px; border-radius: 10px; border-left: 4px solid #ffd700;">
                        <strong style="color: #ffd700;">High Credit Limit</strong>
                        <p style="color: #8a8a9a; margin: 5px 0 0 0;">Credit limit exceeds $10,000</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                if payments < 500:
                    st.markdown("""
                    <div style="background: #ff444420; padding: 15px 20px; border-radius: 10px; border-left: 4px solid #ff4444;">
                        <strong style="color: #ff4444;">Low Payment</strong>
                        <p style="color: #8a8a9a; margin: 5px 0 0 0;">Payments below $500 - need attention</p>
                    </div>
                    """, unsafe_allow_html=True)
                if balance > 10000:
                    st.markdown("""
                    <div style="background: #ff00aa20; padding: 15px 20px; border-radius: 10px; border-left: 4px solid #ff00aa;">
                        <strong style="color: #ff00aa;">High Outstanding Balance</strong>
                        <p style="color: #8a8a9a; margin: 5px 0 0 0;">Balance exceeds $10,000 - risk indicator</p>
                    </div>
                    """, unsafe_allow_html=True)

# ======================
# FOOTER
# ======================
st.markdown("""
<div style="text-align: center; padding: 30px 0; color: #5a5a6a; border-top: 1px solid #2a2a3a; margin-top: 50px;">
    <p style="font-size: 0.8rem; letter-spacing: 0.2em; text-transform: uppercase;">
        ◈ Neural Analytics | Customer Segmentation System
    </p>
</div>
""", unsafe_allow_html=True)
