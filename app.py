# STREAMLIT APP: Retail Clustering
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Retail Clustering Dashboard", layout="wide")
st.title("Retail Customer Clustering Using Spectral, KMeans, and Agglomerative Clustering")

# Upload Dataset
uploaded_file = st.file_uploader("Upload Online Retail II CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    
    # Preprocessing
    st.subheader("Data Preprocessing")
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    df = df.dropna(subset=['Customer ID'])
    df = df[~df['Description'].str.contains('POSTAGE|DISCOUNT|ADJUSTMENT|M|DOTCOM', na=False)]
    st.write(f"Rows after cleaning: {len(df)}")

    # Feature Engineering
    st.subheader("Feature Engineering: Customer-Product Matrix")
    customer_product = pd.crosstab(df['Customer ID'], df['StockCode'])
    customer_product = customer_product.loc[customer_product.sum(axis=1) >= 20]
    customer_product = customer_product.loc[:, customer_product.sum(axis=0) >= 50]

    # Sample subset for clustering
    customer_sample = customer_product.sample(n=2000, random_state=42, axis=0) if customer_product.shape[0] > 2000 else customer_product
    customer_sample = customer_sample.sample(n=500, random_state=42, axis=1) if customer_sample.shape[1] > 500 else customer_sample
    st.write(f"Sampled matrix shape: {customer_sample.shape}")

    X = normalize(customer_sample, norm="l2")

    # Dimensionality reduction
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X)


    # Clustering
    st.subheader("Clustering")
    n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=6, step=1)

    spectral_norm = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors",
                                       assign_labels="kmeans", random_state=42)
    labels_spectral_norm = spectral_norm.fit_predict(X_reduced)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_reduced)

    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    labels_agglo = agglo.fit_predict(X_reduced)

    # Evaluation
    st.subheader("Evaluation Metrics")
    methods = {
        "Spectral (Normalized)": labels_spectral_norm,
        "KMeans": labels_kmeans,
        "Agglomerative": labels_agglo
    }

    eval_results = []
    for name, labels in methods.items():
        sil = silhouette_score(X_reduced, labels)
        db = davies_bouldin_score(X_reduced, labels)
        ch = calinski_harabasz_score(X_reduced, labels)
        eval_results.append({"Method": name, "Silhouette": sil, "Davies-Bouldin": db, "Calinski-Harabasz": ch, "Cluster Sizes": np.bincount(labels)})
    eval_df = pd.DataFrame(eval_results)
    st.dataframe(eval_df)

    # Visualizations
    st.subheader("Visualizations")

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_reduced)

    fig, axes = plt.subplots(1,3,figsize=(18,5))
    for ax, (name, labels) in zip(axes, methods.items()):
        ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10", s=10)
        ax.set_title(f"{name} (PCA 2D)")
    st.pyplot(fig)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    X_tsne = tsne.fit_transform(X_reduced)

    fig2, axes2 = plt.subplots(1,3,figsize=(18,5))
    for ax, (name, labels) in zip(axes2, methods.items()):
        ax.scatter(X_tsne[:,0], X_tsne[:,1], c=labels, cmap="tab10", s=10)
        ax.set_title(f"{name} (t-SNE 2D)")
    st.pyplot(fig2)

    # Cluster size distribution
    st.subheader("Cluster Size Distribution")
    fig3, ax3 = plt.subplots(figsize=(10,5))
    for name, labels in methods.items():
        sns.histplot(labels, bins=n_clusters, stat="count", label=name, alpha=0.5, ax=ax3)
    ax3.legend()
    ax3.set_title("Cluster Size Distribution")
    st.pyplot(fig3)
