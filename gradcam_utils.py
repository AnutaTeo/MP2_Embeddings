import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_gradcam_transform():
    """
    This must match the transform used during classifier training.
    """

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def generate_gradcam(model, image_path, device):
    """
    Generates a Grad-CAM heatmap for the uploaded image.

    Returns:
        visualization: RGB numpy image with heatmap overlay
        predicted_class: integer class predicted by the classifier
    """

    model.eval()

    image = Image.open(image_path).convert("RGB")

    # For visualization, keep an unnormalized RGB image in [0, 1]
    resized_image = image.resize((224, 224))
    rgb_img = np.array(resized_image).astype(np.float32) / 255.0

    # For model input, use normalized tensor
    transform = get_gradcam_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Last convolutional layer in UltraLightCNN
    # UltraLightCNN.features:
    # 0 Conv2d
    # 1 ReLU
    # 2 MaxPool
    # 3 Conv2d  <- target layer
    # 4 ReLU
    # 5 MaxPool
    target_layers = [model.features[3]]

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(predicted_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
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