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
        #Loads embeddings, labels, image paths and the CNN feature extractor

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

        print("Search engine loaded successfully.")
        print("Embeddings shape:", self.embeddings.shape)
        print("Using device:", self.device)

    def search_similar_images(self, query_image_path, top_k=5):
        #Receives one image and returns the top_k most similar images from the dataset

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
            result = {
                "image_path": self.image_paths[index],
                "label": self.labels[index],
                "similarity": float(similarities[index])
            }

            results.append(result)

        prediction = self.predict_from_neighbors(results)

        return results, prediction

    def predict_from_neighbors(self, results):
        #Estimates the class based on the labels of the most similar images (similar to k-nearest neighbors)

        normal_count = 0
        pneumonia_count = 0

        for result in results:
            if result["label"] == "NORMAL":
                normal_count += 1
            elif result["label"] == "PNEUMONIA":
                pneumonia_count += 1

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

        prediction = {
            "predicted_class": predicted_class,
            "normal_count": normal_count,
            "pneumonia_count": pneumonia_count,
            "confidence": confidence
        }

        return prediction