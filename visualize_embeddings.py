import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

#print the shape and first vector
embeddings = np.load("embeddings/embeddings.npy")
image_paths = np.load("embeddings/image_paths.npy")
labels = np.load("embeddings/labels.npy")

print("Embeddings shape:", embeddings.shape)
print("Image paths shape:", image_paths.shape)
print("Labels shape:", labels.shape)

print("\nFirst image path:")
print(image_paths[0])

print("\nFirst label:")
print(labels[0])

print("\nFirst embedding vector:")
print(embeddings[0])

print("\nFirst embedding vector length:")
print(len(embeddings[0]))


#Visualize embeddings in 2D using PCA
#Eeach dot is a radiography
#dots that are close together have similar CNN embeddings, so the images are visually similar

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

normal_points = labels == "NORMAL"
pneumonia_points = labels == "PNEUMONIA"

plt.figure(figsize=(8, 6))

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

plt.title("2D Visualization of X-ray Image Embeddings using PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()



#Visualize embeddings in 3D using PCA

pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

normal_points = labels == "NORMAL"
pneumonia_points = labels == "PNEUMONIA"

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    embeddings_3d[normal_points, 0],
    embeddings_3d[normal_points, 1],
    embeddings_3d[normal_points, 2],
    label="NORMAL",
    alpha=0.7
)

ax.scatter(
    embeddings_3d[pneumonia_points, 0],
    embeddings_3d[pneumonia_points, 1],
    embeddings_3d[pneumonia_points, 2],
    label="PNEUMONIA",
    alpha=0.7
)

ax.set_title("3D PCA Visualization of X-ray Image Embeddings")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
ax.legend()

plt.show()



#Visualize embeddings in 5D using PCA
#Component 4: color
#Component 5: point size

pca = PCA(n_components=5)
embeddings_5d = pca.fit_transform(embeddings)

x = embeddings_5d[:, 0]
y = embeddings_5d[:, 1]
z = embeddings_5d[:, 2]
color_value = embeddings_5d[:, 3]
size_value = embeddings_5d[:, 4]

# Normalize point sizes so they are visible
size_value = size_value - size_value.min()
size_value = 30 + 200 * (size_value / size_value.max())

normal_points = labels == "NORMAL"
pneumonia_points = labels == "PNEUMONIA"

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

normal_plot = ax.scatter(
    x[normal_points],
    y[normal_points],
    z[normal_points],
    c=color_value[normal_points],
    s=size_value[normal_points],
    marker="o",
    alpha=0.7,
    label="NORMAL"
)

pneumonia_plot = ax.scatter(
    x[pneumonia_points],
    y[pneumonia_points],
    z[pneumonia_points],
    c=color_value[pneumonia_points],
    s=size_value[pneumonia_points],
    marker="^",
    alpha=0.7,
    label="PNEUMONIA"
)

ax.set_title("5D PCA Visualization of X-ray Embeddings")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")

fig.colorbar(normal_plot, ax=ax, label="PCA Component 4")
ax.legend()

plt.show()


#visualize nearest neighbors and 5 images that are similar
query_index = 0
top_k = 5

query_embedding = embeddings[query_index].reshape(1, -1)

similarities = cosine_similarity(query_embedding, embeddings)[0]

# Exclude the same image from the results
similarities[query_index] = -1

top_indices = similarities.argsort()[::-1][:top_k]

fig, axes = plt.subplots(1, top_k + 1, figsize=(18, 4))

query_image = Image.open(image_paths[query_index])
axes[0].imshow(query_image, cmap="gray")
axes[0].set_title(f"Query\n{labels[query_index]}")
axes[0].axis("off")

for position, index in enumerate(top_indices, start=1):
    image = Image.open(image_paths[index])

    axes[position].imshow(image, cmap="gray")
    axes[position].set_title(
        f"Top {position}\n{labels[index]}\nSim: {similarities[index]:.3f}"
    )
    axes[position].axis("off")

plt.tight_layout()
plt.show()
