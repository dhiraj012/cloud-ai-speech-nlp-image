from transformers import pipeline

def label_image(image_path="photo.jpg"):
    model = pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )

    results = model(image_path, top_k=5)

    labels = []

    for item in results:
        labels.append({
            "label": item["label"],
            "confidence": round(float(item["score"]), 4)
        })

    return labels

if __name__ == "__main__":
    results = label_image()

    print("Top 5 Image Labels:")

    for item in results:
        print(item)
