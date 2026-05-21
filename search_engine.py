import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
EMBEDDINGS_PATH = "embeddings/embeddings.npy"
IMAGE_PATHS_PATH = "embeddings/image_paths.npy"
LABELS_PATH = "embeddings/labels.npy"

# Model loading
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_feature_extractor(device):
    print("Loading pretrained DenseNet121 model...")

    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)

    feature_extractor = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d((1, 1))
    )

    feature_extractor.to(device)
    feature_extractor.eval()

    return feature_extractor


def get_image_transform():
    #Transform image to the format expected
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_embedding(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image_tensor)

    embedding = embedding.squeeze().cpu().numpy()

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


# Search engine class
class XRaySearchEngine:
    def __init__(self):
        if not os.path.exists(EMBEDDINGS_PATH):
            raise FileNotFoundError("embeddings.npy not found. Run extract_embeddings.py first.")

        if not os.path.exists(IMAGE_PATHS_PATH):
            raise FileNotFoundError("image_paths.npy not found. Run extract_embeddings.py first.")

        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError("labels.npy not found. Run extract_embeddings.py first.")

        self.embeddings = np.load(EMBEDDINGS_PATH)
        self.image_paths = np.load(IMAGE_PATHS_PATH)
        self.labels = np.load(LABELS_PATH)

        self.device = get_device()
        self.model = load_feature_extractor(self.device)
        self.transform = get_image_transform()

        print("Search engine loaded.")
        print("Embeddings shape:", self.embeddings.shape)
        print("Device:", self.device)

    def search_similar_images(self, query_image_path, top_k=5):
        query_embedding = extract_embedding(
            query_image_path,
            self.model,
            self.transform,
            self.device
        )

        query_embedding = query_embedding.reshape(1, -1)

        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []

        for index in top_indices:
            results.append({
                "image_path": str(self.image_paths[index]),
                "label": str(self.labels[index]),
                "similarity": float(similarities[index])
            })

        prediction = self.predict_from_neighbors(results)
        explanation = self.create_explanation(results, prediction)

        return results, prediction, explanation, query_embedding.flatten()

    def predict_from_neighbors(self, results):
        normal_count = sum(1 for r in results if r["label"] == "NORMAL")
        pneumonia_count = sum(1 for r in results if r["label"] == "PNEUMONIA")
        total = len(results)

        if pneumonia_count > normal_count:
            predicted_class = "PNEUMONIA"
            confidence = pneumonia_count / total
        elif normal_count > pneumonia_count:
            predicted_class = "NORMAL"
            confidence = normal_count / total
        else:
            predicted_class = "UNCERTAIN"
            confidence = 0.5

        return {
            "predicted_class": predicted_class,
            "normal_count": normal_count,
            "pneumonia_count": pneumonia_count,
            "confidence": confidence
        }

    def create_explanation(self, results, prediction):
        normal_scores = [r["similarity"] for r in results if r["label"] == "NORMAL"]
        pneumonia_scores = [r["similarity"] for r in results if r["label"] == "PNEUMONIA"]

        avg_normal = sum(normal_scores) / len(normal_scores) if normal_scores else 0
        avg_pneumonia = sum(pneumonia_scores) / len(pneumonia_scores) if pneumonia_scores else 0

        return {
            "avg_normal_similarity": avg_normal,
            "avg_pneumonia_similarity": avg_pneumonia,
            "text": (
                f"The system found {prediction['normal_count']} NORMAL and "
                f"{prediction['pneumonia_count']} PNEUMONIA images among the closest matches. "
                f"The final estimation is based on majority voting over the nearest neighbors."
            )
        }