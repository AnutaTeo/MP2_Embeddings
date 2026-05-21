import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Paths
# =========================

EMBEDDINGS_PATH = "embeddings/embeddings.npy"
IMAGE_PATHS_PATH = "embeddings/image_paths.npy"
LABELS_PATH = "embeddings/labels.npy"
CLASSIFIER_PATH = "classifier_model.pth"


# =========================
# Small classifier used for Grad-CAM
# Must match train_classifier.py
# =========================

class UltraLightCNN(nn.Module):
    def __init__(self):
        super(UltraLightCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 224 -> 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 112 -> 56
        )

        self.classifier = nn.Linear(32 * 56 * 56, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# =========================
# Device and transforms
# =========================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_image_transform():
    """
    This transform must be the same as the one used in extract_embeddings.py.
    DenseNet121 expects 224x224 normalized RGB images.
    """

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =========================
# DenseNet121 feature extractor
# =========================

def load_feature_extractor(device):
    """
    Loads DenseNet121 pretrained on ImageNet and removes the final classifier.
    Output embedding size: 1024.
    """

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


def extract_embedding(image_path, model, transform, device):
    """
    Converts an image into a normalized DenseNet121 embedding.
    """

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image_tensor)

    embedding = embedding.squeeze().cpu().numpy()

    # L2 normalization
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


# =========================
# Search Engine
# =========================

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

        print("Loading UltraLightCNN classifier for Grad-CAM...")

        self.classifier = UltraLightCNN()

        if os.path.exists(CLASSIFIER_PATH):
            self.classifier.load_state_dict(
                torch.load(CLASSIFIER_PATH, map_location=self.device)
            )
            print("Classifier loaded from classifier_model.pth.")
        else:
            print("Warning: classifier_model.pth not found. Grad-CAM may not work correctly.")

        self.classifier.to(self.device)
        self.classifier.eval()

        print("Search engine loaded.")
        print("Embeddings shape:", self.embeddings.shape)
        print("Device:", self.device)

    def search_similar_images(self, query_image_path, top_k=5, similarity_threshold=0.60):
        """
        Returns exactly 4 values:

        results, prediction, explanation, query_embedding

        If the best similarity is below the threshold, the input is treated as
        not similar enough to the chest X-ray dataset.
        """

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

        best_similarity = results[0]["similarity"]

        # Threshold validation
        if best_similarity < similarity_threshold:
            prediction = {
                "predicted_class": "NOT_CHEST_XRAY",
                "normal_count": 0,
                "pneumonia_count": 0,
                "confidence": 0,
                "best_similarity": best_similarity,
                "threshold": similarity_threshold,
                "is_valid_xray": False
            }

            explanation = {
                "avg_normal_similarity": 0,
                "avg_pneumonia_similarity": 0,
                "text": (
                    f"The uploaded image is not similar enough to the chest X-ray dataset. "
                    f"The highest similarity score was {best_similarity:.3f}, which is below "
                    f"the threshold of {similarity_threshold:.2f}. Therefore, the system does "
                    f"not estimate NORMAL or PNEUMONIA."
                )
            }

            return results, prediction, explanation, query_embedding.flatten()

        prediction = self.predict_from_neighbors(results)

        prediction["best_similarity"] = best_similarity
        prediction["threshold"] = similarity_threshold
        prediction["is_valid_xray"] = True

        explanation = self.create_explanation(results, prediction)

        return results, prediction, explanation, query_embedding.flatten()

    def predict_from_neighbors(self, results):
        """
        Majority vote over nearest neighbors.
        """

        normal_count = sum(1 for result in results if result["label"] == "NORMAL")
        pneumonia_count = sum(1 for result in results if result["label"] == "PNEUMONIA")

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
        normal_scores = [
            result["similarity"]
            for result in results
            if result["label"] == "NORMAL"
        ]

        pneumonia_scores = [
            result["similarity"]
            for result in results
            if result["label"] == "PNEUMONIA"
        ]

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