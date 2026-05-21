import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import plotly.express as px


EMBEDDINGS_PATH = "embeddings/embeddings.npy"
LABELS_PATH = "embeddings/labels.npy"

VISUALIZATION_FOLDER = "static/visualizations"


def load_embeddings_and_labels():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError("embeddings.npy not found. Run extract_embeddings.py first.")

    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError("labels.npy not found. Run extract_embeddings.py first.")

    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)

    return embeddings, labels


def generate_2d_pca_plot(embeddings, labels):
    """
    Static 2D PCA image.
    Each point represents one X-ray image embedding.
    """

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    normal_points = labels == "NORMAL"
    pneumonia_points = labels == "PNEUMONIA"

    plt.figure(figsize=(9, 7))

    plt.scatter(
        embeddings_2d[normal_points, 0],
        embeddings_2d[normal_points, 1],
        label="NORMAL",
        alpha=0.7
    )

    plt.scatter(
        embeddings_2d[pneumonia_points, 0],
        embeddings_2d[pneumonia_points, 1],
        label="PNEUMONIA",
        alpha=0.7
    )

    plt.title("2D PCA Visualization of X-ray Embeddings")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(VISUALIZATION_FOLDER, "pca_2d.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def generate_interactive_3d_pca_plot(embeddings, labels):
    #Interactive 3D PCA plot

    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)

    df = pd.DataFrame({
        "PCA 1": embeddings_3d[:, 0],
        "PCA 2": embeddings_3d[:, 1],
        "PCA 3": embeddings_3d[:, 2],
        "Class": labels,
        "Image Index": list(range(len(labels)))
    })

    fig = px.scatter_3d(
        df,
        x="PCA 1",
        y="PCA 2",
        z="PCA 3",
        color="Class",
        hover_data=["Image Index", "Class"],
        title="Interactive 3D PCA Visualization of X-ray Embeddings",
        opacity=0.75
    )

    fig.update_traces(marker=dict(size=4))

    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            zaxis_title="PCA Component 3"
        )
    )

    output_path = os.path.join(VISUALIZATION_FOLDER, "pca_3d_interactive.html")

    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True
    )

    return output_path


def generate_all_visualizations():
    os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

    embeddings, labels = load_embeddings_and_labels()

    generate_2d_pca_plot(embeddings, labels)
    generate_interactive_3d_pca_plot(embeddings, labels)

    return {
        "pca_2d": "visualizations/pca_2d.png",
        "pca_3d_interactive": "visualizations/pca_3d_interactive.html",
        "total_images": len(labels),
        "embedding_size": embeddings.shape[1],
        "normal_count": int(np.sum(labels == "NORMAL")),
        "pneumonia_count": int(np.sum(labels == "PNEUMONIA"))
    }