import os
from pathlib import Path

import kagglehub
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# Configuration
DATASET_NAME = "paultimothymooney/chest-xray-pneumonia"
OUTPUT_DIR = "embeddings"
CLASSES = ["NORMAL", "PNEUMONIA"]

MAX_IMAGES_PER_CLASS = 3000


# Dataset download
def download_dataset():
    print("Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download(DATASET_NAME)

    print("Path to dataset files:", path)
    return Path(path)


def find_chest_xray_folder(dataset_root):
    dataset_root = Path(dataset_root)

    for folder in dataset_root.rglob("*"):
        normal_path = folder / "train" / "NORMAL"
        pneumonia_path = folder / "train" / "PNEUMONIA"

        if normal_path.exists() and pneumonia_path.exists():
            print("Found chest_xray folder:", folder)
            return folder

    raise FileNotFoundError(
        "Could not find the chest_xray folder with train/NORMAL and train/PNEUMONIA."
    )



# Model and preprocessing
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_feature_extractor(device):
    #Loads a pretrained DenseNet121 model and removes the final classification layer
    #DenseNet is commonly used for medical image feature extraction

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


# Image collection
def collect_image_paths(chest_xray_folder):
    #Collects image paths from: train/NORMAL and train/PNEUMONIA

    image_paths = []
    labels = []

    train_folder = chest_xray_folder / "train"

    for class_name in CLASSES:
        class_folder = train_folder / class_name

        if not class_folder.exists():
            raise FileNotFoundError(f"Folder not found: {class_folder}")

        all_images = [
            file for file in class_folder.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        selected_images = all_images[:MAX_IMAGES_PER_CLASS]

        print(f"{class_name}: using {len(selected_images)} images")

        for image_path in selected_images:
            image_paths.append(str(image_path))
            labels.append(class_name)

    return image_paths, labels

# Main
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset_root = download_dataset()
    chest_xray_folder = find_chest_xray_folder(dataset_root)

    device = get_device()
    print("Using device:", device)

    model = load_feature_extractor(device)
    transform = get_image_transform()

    image_paths, labels = collect_image_paths(chest_xray_folder)

    print(f"Total images selected: {len(image_paths)}")
    print("Extracting embeddings...")

    embeddings = []
    valid_image_paths = []
    valid_labels = []

    for index, image_path in enumerate(image_paths):
        try:
            embedding = extract_embedding(image_path, model, transform, device)

            embeddings.append(embedding)
            valid_image_paths.append(image_path)
            valid_labels.append(labels[index])

            if (index + 1) % 25 == 0:
                print(f"Processed {index + 1}/{len(image_paths)} images")

        except Exception as error:
            print("Error processing image:", image_path)
            print(error)

    embeddings = np.array(embeddings)
    valid_image_paths = np.array(valid_image_paths)
    valid_labels = np.array(valid_labels)

    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    np.save(os.path.join(OUTPUT_DIR, "image_paths.npy"), valid_image_paths)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), valid_labels)

    print("\nDone.")
    print("Embeddings shape:", embeddings.shape)
    print("Saved files:")
    print("-", os.path.join(OUTPUT_DIR, "embeddings.npy"))
    print("-", os.path.join(OUTPUT_DIR, "image_paths.npy"))
    print("-", os.path.join(OUTPUT_DIR, "labels.npy"))


if __name__ == "__main__":
    main()