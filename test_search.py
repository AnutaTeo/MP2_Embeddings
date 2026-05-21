import matplotlib.pyplot as plt
from PIL import Image

from search_engine import XRaySearchEngine


def show_results(query_image_path, results, prediction, explanation):
    top_k = len(results)

    fig, axes = plt.subplots(1, top_k + 2, figsize=(22, 4))

    query_image = Image.open(query_image_path)
    axes[0].imshow(query_image, cmap="gray")
    axes[0].set_title("Query Image")
    axes[0].axis("off")
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    for position, result in enumerate(results, start=2):
        image = Image.open(result["image_path"])

        axes[position].imshow(image, cmap="gray")
        axes[position].set_title(
            f"Top {position}\n"
            f"{result['label']}\n"
            f"Sim: {result['similarity']:.3f}"
        )
        axes[position].axis("off")

    plt.suptitle(
        f"Prediction: {prediction['predicted_class']} | "
        f"Confidence: {prediction['confidence']:.2f} | "
        f"Normal: {prediction['normal_count']} | "
        f"Pneumonia: {prediction['pneumonia_count']}\n"
        f"Avg NORMAL sim: {explanation['avg_normal_similarity']:.3f} | "
        f"Avg PNEUMONIA sim: {explanation['avg_pneumonia_similarity']:.3f}"
    )

    plt.tight_layout()
    plt.show()


def main():
    search_engine = XRaySearchEngine()

    # Change this to your real image path
    query_image_path = r"C:\Users\User\Desktop\Prezentari\MP2_Embeddings\image3_n.jpeg"

    results, prediction, explanation, query_embedding = search_engine.search_similar_images(
        query_image_path=query_image_path,
        top_k=5
    )

    print("\nPrediction:")
    print(prediction)

    print("\nExplanation:")
    print(explanation)

    print("\nQuery embedding size:")
    print(len(query_embedding))

    print("\nFirst 10 embedding values:")
    print(query_embedding[:10])

    print("\nTop similar images:")
    for index, result in enumerate(results, start=1):
        print(
            index,
            result["label"],
            round(result["similarity"], 4),
            result["image_path"]
        )

    show_results(results, prediction, explanation, query_embedding, heatmap_path)


if __name__ == "__main__":
    main()