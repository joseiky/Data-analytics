import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import os

# === CONFIG ===
input_file = "Empty_variant_rows_removed.xlsx"
features = ['Variant', 'mRNA', 'Protein', 'Common Name', 'Zygosity']
output_dir = "pca_kmeans_outputs"
os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA ===
df = pd.read_excel(input_file)

# === ENCODER FUNCTION ===
def prepare_data(df, group_col):
    df = df[[group_col] + features].dropna()
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_enc = enc.fit_transform(df[features])
    col_names = enc.get_feature_names_out(features)
    df_enc = pd.DataFrame(X_enc, columns=col_names)
    df_enc[group_col] = df[group_col].values
    df_grouped = df_enc.groupby(group_col).sum().reset_index()
    return df_grouped, enc

# === PCA + CLUSTERING FUNCTION ===
def pca_kmeans_plot(df_grouped, id_col, prefix, k=3):
    X = df_grouped.drop(columns=[id_col])
    ids = df_grouped[id_col]
    
    # === PCA ===
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Scree plot
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(PCA().fit(X).explained_variance_ratio_), marker='o')
    plt.title(f"{prefix} – PCA Scree")
    plt.xlabel("Components")
    plt.ylabel("Cumulative Variance")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_scree.png", dpi=300)
    plt.close()

    # Elbow plot
    inertias = []
    for i in range(2, 7):
        kmeans = KMeans(n_clusters=i, random_state=42).fit(X_pca)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(range(2, 7), inertias, marker='o')
    plt.title(f"{prefix} – Elbow Plot")
    plt.xlabel("Clusters")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_elbow.png", dpi=300)
    plt.close()

    # Final KMeans clustering
    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(X_pca)

    # Static plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="Set2", s=70)
    plt.title(f"{prefix} – PCA + KMeans (k={k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_cluster.png", dpi=600)
    plt.close()

    # Interactive plot
    fig = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=clusters.astype(str),
        hover_name=ids.astype(str),
        title=f"{prefix} – Interactive PCA Clusters",
        labels={"x": "PC1", "y": "PC2", "color": "Cluster"}
    )
    fig.write_html(f"{output_dir}/{prefix}_interactive.html")

# === RUN FOR BOTH GROUPINGS (same column for now, different future versions possible) ===
for group_by, name in [("MDL Patient ID", "By_MDL_Patient_ID"), ("MDL Patient ID", "By_Patient_ID")]:
    try:
        grouped_df, _ = prepare_data(df, group_by)
        pca_kmeans_plot(grouped_df, group_by, name)
    except Exception as e:
        print(f"⚠️ Error in {name}: {e}")

print(f"✅ All plots saved to: {output_dir}/")
