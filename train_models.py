#!/usr/bin/env python3
"""
Train K-Means, KNN, Decision Tree, and Random Forest models
for Credit Card Customer Segmentation
"""

import pandas as pd
import numpy as np
import joblib
import os

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "CC GENERAL.csv")
models_dir = os.path.join(base_dir, "models")

# Create models directory if not exists
os.makedirs(models_dir, exist_ok=True)

print("=" * 60)
print("CREDIT CARD CUSTOMER SEGMENTATION - MODEL TRAINING")
print("=" * 60)

# 1. Load data
print("\n[1] Loading data...")
df = pd.read_csv(data_path)
print(f"   Data shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# 2. Clean data
print("\n[2] Cleaning data...")
df = df.drop_duplicates()
df = df.drop('CUST_ID', axis=1)
df = df.fillna(df.mean(numeric_only=True))
print(f"   After cleaning: {df.shape}")

# 3. Outlier Detection & Removal
print("\n[3] Outlier Detection & Removal...")
features = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']

# Show outlier statistics before
def count_outliers_iqr(series, k=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return ((series < lower) | (series > upper)).sum()

print("   Outliers before removal (IQR method):")
total_outliers = 0
for feat in features:
    outliers = count_outliers_iqr(df[feat])
    total_outliers += outliers
    print(f"     {feat}: {outliers} outliers ({outliers/len(df)*100:.1f}%)")

# Remove outliers using percentile-based clipping (more robust)
def clip_outliers_percentile(df, features, lower=0.02, upper=0.98):
    """Clip outliers to percentile boundaries"""
    df_clipped = df.copy()
    for feat in features:
        low = df[feat].quantile(lower)
        high = df[feat].quantile(upper)
        df_clipped[feat] = df_clipped[feat].clip(low, high)
    return df_clipped

df = clip_outliers_percentile(df, features)
print(f"   Applied percentile clipping (2%, 98%)")

# Show stats after clipping
print("   Data statistics after clipping:")
for feat in features:
    print(f"     {feat}: min={df[feat].min():.2f}, max={df[feat].max():.2f}, mean={df[feat].mean():.2f}")

# 4. Feature selection & scaling
print("\n[4] Feature selection & scaling...")
from sklearn.preprocessing import RobustScaler  # More robust than StandardScaler

X = df[features].copy()

scaler = RobustScaler()  # Uses median/IQR, less sensitive to outliers
X_scaled = scaler.fit_transform(X)
print(f"   Features: {features}")
print(f"   X_scaled shape: {X_scaled.shape}")
print(f"   Using RobustScaler for outlier-resistant scaling")

# 4. K-Means Clustering
print("\n[4] Training K-Means Clustering...")
from sklearn.cluster import KMeans

# Use k=4 based on Elbow Method
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_counts = df['Cluster'].value_counts().sort_index()
print(f"   K-Means trained with k=4")
print(f"   Cluster distribution:")
for cluster_id, count in cluster_counts.items():
    print(f"     Cluster {cluster_id}: {count} customers ({count/len(df)*100:.1f}%)")

# 5. Train/ Test split
print("\n[5] Preparing train/test split...")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df['Cluster'], test_size=0.2, random_state=42
)
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")

# 6. Train KNN
print("\n[6] Training KNN Classifier...")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print(f"   KNN Accuracy: {knn_accuracy*100:.1f}%")

# 7. Train Decision Tree
print("\n[7] Training Decision Tree Classifier...")
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=42, max_depth=10)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_pred)
print(f"   Decision Tree Accuracy: {tree_accuracy*100:.1f}%")

# 8. Train Random Forest
print("\n[8] Training Random Forest Classifier...")
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"   Random Forest Accuracy: {rf_accuracy*100:.1f}%")

# 9. Save models
print("\n[9] Saving models...")
joblib.dump(kmeans, os.path.join(models_dir, "kmeans.pkl"))
joblib.dump(knn, os.path.join(models_dir, "knn.pkl"))
joblib.dump(tree, os.path.join(models_dir, "tree.pkl"))
joblib.dump(rf, os.path.join(models_dir, "rf.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

print(f"   Saved to: {models_dir}/")

# 10. Summary
print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nModels saved:")
print(f"  - kmeans.pkl     (K-Means Clustering)")
print(f"  - knn.pkl        (KNN Classifier)")
print(f"  - tree.pkl       (Decision Tree Classifier)")
print(f"  - rf.pkl         (Random Forest Classifier)")
print(f"  - scaler.pkl     (StandardScaler)")

print("\nModel Performance:")
print(f"  {'Model':<20} {'Accuracy':>10}")
print(f"  {'-'*30}")
print(f"  {'KNN (k=5)':<20} {knn_accuracy*100:>9.1f}%")
print(f"  {'Decision Tree':<20} {tree_accuracy*100:>9.1f}%")
print(f"  {'Random Forest':<20} {rf_accuracy*100:>9.1f}%")
print("=" * 60)