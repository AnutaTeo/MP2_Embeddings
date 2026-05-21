import os
import shutil
from uuid import uuid4

from flask import Flask, render_template, request, url_for
from PIL import Image, ImageOps

from search_engine import XRaySearchEngine
from gradcam_utils import generate_gradcam

# Keep this import if your visualizations route uses this function.
# In your repo, the file is called visualize_embeddings.py.
from visualize_embeddings import generate_all_visualizations


app = Flask(__name__)


# =========================
# Folders
# =========================

UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
PROCESSED_FOLDER = "static/processed"
GRADCAM_FOLDER = "static/gradcam"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)


# =========================
# Load search engine once
# =========================

search_engine = XRaySearchEngine()


# =========================
# Helper functions
# =========================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_preprocessed_preview(image_path, output_path):
    """
    Creates a visual preview of the preprocessing step.
    This is only for the interface.
    The actual model preprocessing happens in search_engine.py.
    """

    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image)
    image = image.resize((224, 224))
    image.save(output_path)


def copy_result_image(original_path, result_index):
    """
    Copies dataset images from Kaggle cache/local paths into static/results
    so Flask can display them in the browser.
    """

    extension = os.path.splitext(original_path)[1]

    if extension == "":
        extension = ".jpg"

    new_filename = f"result_{result_index}_{uuid4().hex}{extension}"
    destination = os.path.join(RESULTS_FOLDER, new_filename)

    shutil.copy(original_path, destination)

    return f"results/{new_filename}"


def save_gradcam_image(heatmap_array):
    """
    Saves Grad-CAM RGB numpy image into static/gradcam.
    Does not require cv2.
    """

    gradcam_filename = f"gradcam_{uuid4().hex}.jpg"
    gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)

    Image.fromarray(heatmap_array).save(gradcam_path)

    return url_for("static", filename=f"gradcam/{gradcam_filename}")


# =========================
# Routes
# =========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "xray_image" not in request.files:
        return render_template("index.html", error="No file uploaded.")

    file = request.files["xray_image"]

    if file.filename == "":
        return render_template("index.html", error="No selected file.")

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Only PNG, JPG, and JPEG files are allowed."
        )

    # -------------------------
    # Save uploaded image
    # -------------------------

    unique_name = f"{uuid4().hex}_{file.filename}"
    upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(upload_path)

    # -------------------------
    # Save preprocessing preview
    # -------------------------

    processed_name = f"processed_{unique_name}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_name)
    save_preprocessed_preview(upload_path, processed_path)

    # -------------------------
    # Similarity search
    # -------------------------

    top_k = int(request.form.get("top_k", 5))

    results, prediction, explanation, query_embedding = search_engine.search_similar_images(
        query_image_path=upload_path,
        top_k=top_k,
        similarity_threshold=0.60
    )

    # -------------------------
    # Grad-CAM
    # Only generate if image is similar enough to chest X-ray dataset.
    # -------------------------

    gradcam_url = None
    gradcam_predicted_class = None

    if prediction.get("is_valid_xray", False):
        heatmap, gradcam_predicted_class = generate_gradcam(
            search_engine.classifier,
            upload_path,
            search_engine.device
        )

        gradcam_url = save_gradcam_image(heatmap)

    # -------------------------
    # Copy search results for display
    # -------------------------

    browser_results = []

    for index, result in enumerate(results, start=1):
        static_result_path = copy_result_image(result["image_path"], index)

        browser_results.append({
            "rank": index,
            "image_url": url_for("static", filename=static_result_path),
            "label": result["label"],
            "similarity": round(result["similarity"], 4),
            "original_path": result["image_path"]
        })

    # -------------------------
    # Embedding preview
    # -------------------------

    embedding_preview = query_embedding[:12].round(4).tolist()

    # -------------------------
    # Percentages for bar chart
    # -------------------------

    total_neighbors = prediction["normal_count"] + prediction["pneumonia_count"]

    if total_neighbors > 0:
        normal_percentage = round(
            (prediction["normal_count"] / total_neighbors) * 100,
            2
        )

        pneumonia_percentage = round(
            (prediction["pneumonia_count"] / total_neighbors) * 100,
            2
        )
    else:
        normal_percentage = 0
        pneumonia_percentage = 0

    # -------------------------
    # Visual pipeline steps
    # -------------------------

    process_steps = [
        {
            "title": "1. Image Upload",
            "description": "The user uploads a chest X-ray image from the computer."
        },
        {
            "title": "2. Preprocessing",
            "description": "The image is converted to RGB, resized to 224x224, transformed into a tensor, and normalized."
        },
        {
            "title": "3. CNN Feature Extraction",
            "description": "DenseNet121 extracts a numerical image embedding from the uploaded radiograph."
        },
        {
            "title": "4. Similarity Search",
            "description": "The image embedding is compared with all stored dataset embeddings using cosine similarity."
        },
        {
            "title": "5. Threshold Validation",
            "description": "If the best similarity is below 0.60, the input is rejected as not similar enough to the chest X-ray dataset."
        },
        {
            "title": "6. Neighbor Voting",
            "description": "For valid chest X-rays, the final estimation is based on the labels of the most similar images."
        },
        {
            "title": "7. Grad-CAM Explainability",
            "description": "Grad-CAM shows which image regions influenced the classifier prediction."
        }
    ]

    return render_template(
        "results.html",
        uploaded_image=url_for("static", filename=f"uploads/{unique_name}"),
        processed_image=url_for("static", filename=f"processed/{processed_name}"),
        gradcam_image=gradcam_url,
        gradcam_predicted_class=gradcam_predicted_class,
        results=browser_results,
        prediction=prediction,
        explanation=explanation,
        embedding_preview=embedding_preview,
        embedding_size=len(query_embedding),
        process_steps=process_steps,
        normal_percentage=normal_percentage,
        pneumonia_percentage=pneumonia_percentage,
        total_neighbors=total_neighbors
    )


@app.route("/visualizations")
def visualizations():
    visualization_data = generate_all_visualizations()

    return render_template(
        "visualizations.html",
        visualization_data=visualization_data
    )


if __name__ == "__main__":
    app.run(debug=True)