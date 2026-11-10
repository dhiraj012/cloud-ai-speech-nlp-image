from transformers import pipeline

def analyze_sentiment(text):
    model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    result = model(text[:512])[0]

    return {
        "sentiment": result["label"].lower(),
        "score": round(float(result["score"]), 4)
    }

if __name__ == "__main__":
    text = "The weather is beautiful and I feel happy."
    print(analyze_sentiment(text))
