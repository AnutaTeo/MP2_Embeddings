import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_gradcam(model, image_path, device):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")

    rgb_img = np.array(
        image.resize((224, 224))
    ) / 255.0

    input_tensor = transform(image).unsqueeze(0).to(device)

    # last DenseNet conv layer
    target_layers = [model.features[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    output = model(input_tensor)

    predicted_class = output.argmax(dim=1).item()

    targets = [
        ClassifierOutputTarget(predicted_class)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return visualization, predicted_class