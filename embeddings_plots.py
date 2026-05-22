import os
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDING_ANALYSIS_FOLDER = "static/embedding_analysis"


def generate_embedding_analysis_plots(
    query_embedding,
    dataset_embeddings,
    dataset_labels,
    similarity_threshold=0.60
):
    os.makedirs(EMBEDDING_ANALYSIS_FOLDER, exist_ok=True)

    unique_id = uuid4().hex

    query_embedding_2d = query_embedding.reshape(1, -1)

    similarities = cosine_similarity(query_embedding_2d, dataset_embeddings)[0]

    similarity_plot = generate_similarity_distribution_plot(
        similarities,
        similarity_threshold,
        unique_id
    )

    dimensions_plot = generate_top_embedding_dimensions_plot(
        query_embedding,
        unique_id
    )

    pca_plot = generate_query_pca_position_plot(
        query_embedding,
        dataset_embeddings,
        dataset_labels,
        unique_id
    )

    return {
        "similarity_distribution": similarity_plot,
        "top_embedding_dimensions": dimensions_plot,
        "query_pca_position": pca_plot
    }


def generate_similarity_distribution_plot(similarities, similarity_threshold, unique_id):
    output_filename = f"similarity_distribution_{unique_id}.png"
    output_path = os.path.join(EMBEDDING_ANALYSIS_FOLDER, output_filename)

    best_similarity = np.max(similarities)
    mean_similarity = np.mean(similarities)

    plt.figure(figsize=(9, 5))
    plt.hist(similarities, bins=30, alpha=0.8)

    plt.axvline(
        best_similarity,
        linestyle="--",
        linewidth=2,
        label=f"Best similarity: {best_similarity:.3f}"
    )

    plt.axvline(
        mean_similarity,
        linestyle="--",
        linewidth=2,
        label=f"Mean similarity: {mean_similarity:.3f}"
    )

    plt.axvline(
        similarity_threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {similarity_threshold:.2f}"
    )

    plt.title("Similarity Distribution Against Dataset")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Number of Dataset Images")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return f"embedding_analysis/{output_filename}"


def generate_top_embedding_dimensions_plot(query_embedding, unique_id):
    output_filename = f"top_embedding_dimensions_{unique_id}.png"
    output_path = os.path.join(EMBEDDING_ANALYSIS_FOLDER, output_filename)

    absolute_values = np.abs(query_embedding)

    top_indices = np.argsort(absolute_values)[::-1][:20]
    top_values = query_embedding[top_indices]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(top_indices)), top_values)

    plt.title("Top 20 Strongest Embedding Dimensions")
    plt.xlabel("Embedding Dimension Index")
    plt.ylabel("Activation Value")
    plt.xticks(
        range(len(top_indices)),
        top_indices,
        rotation=60
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return f"embedding_analysis/{output_filename}"


def generate_query_pca_position_plot(query_embedding, dataset_embeddings, dataset_labels, unique_id):
    output_filename = f"query_pca_position_{unique_id}.png"
    output_path = os.path.join(EMBEDDING_ANALYSIS_FOLDER, output_filename)

    combined_embeddings = np.vstack([
        dataset_embeddings,
        query_embedding.reshape(1, -1)
    ])

    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined_embeddings)

    dataset_2d = combined_2d[:-1]
    query_2d = combined_2d[-1]

    normal_points = dataset_labels == "NORMAL"
    pneumonia_points = dataset_labels == "PNEUMONIA"

    plt.figure(figsize=(9, 7))

    plt.scatter(
        dataset_2d[normal_points, 0],
        dataset_2d[normal_points, 1],
        label="NORMAL dataset images",
        alpha=0.5
    )

    plt.scatter(
        dataset_2d[pneumonia_points, 0],
        dataset_2d[pneumonia_points, 1],
        label="PNEUMONIA dataset images",
        alpha=0.5
    )

    plt.scatter(
        query_2d[0],
        query_2d[1],
        marker="X",
        s=250,
        label="Uploaded image",
        edgecolors="black"
    )

    plt.title("Uploaded Image Position in PCA Embedding Space")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return f"embedding_analysis/{output_filename}"