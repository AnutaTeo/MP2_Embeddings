import matplotlib.pyplot as plt
from PIL import Image

from search_engine import XRaySearchEngine


def show_results(query_image_path, results, prediction):
    top_k = len(results)

    fig, axes = plt.subplots(1, top_k + 1, figsize=(18, 4))

    query_image = Image.open(query_image_path)
    axes[0].imshow(query_image, cmap="gray")
    axes[0].set_title("Query Image")
    axes[0].axis("off")

    for position, result in enumerate(results, start=1):
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
        f"Pneumonia: {prediction['pneumonia_count']}"
    )

    plt.tight_layout()
    plt.show()


def main():
    search_engine = XRaySearchEngine()

    query_image_path = r"C:\Users\User\Desktop\Prezentari\MP2_Embeddings\image3_n.jpeg"

    results, prediction = search_engine.search_similar_images(
        query_image_path=query_image_path,
        top_k=10
    )

    print("\nPrediction:")
    print(prediction)

    print("\nTop similar images:")
    for index, result in enumerate(results, start=1):
        print(
            index,
            result["label"],
            result["similarity"],
            result["image_path"]
        )

    show_results(query_image_path, results, prediction)


if __name__ == "__main__":
    main()