import json
from datetime import datetime, timezone

from speech import transcribe_audio
from nlp import analyze_sentiment
from image import label_image

def save_log(service, input_data, output_data):

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": service,
        "input": input_data,
        "output": output_data
    }

    with open("api_calls.log", "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\\n")


def run_pipeline():

    print("\\n--- SPEECH TO TEXT ---")

    text = transcribe_audio("audio.wav")
    print("Transcription:", text)

    save_log("speech", "audio.wav", text)

    print("\\n--- SENTIMENT ANALYSIS ---")

    sentiment = analyze_sentiment(text)
    print("Sentiment:", sentiment)

    save_log("nlp", text, sentiment)

    print("\\n--- IMAGE LABELLING ---")

    labels = label_image("photo.jpg")
    print("Image Labels:", labels)

    save_log("image", "photo.jpg", {"labels": labels})

    print("\\nAll tasks completed successfully!")
    print("Results saved in api_calls.log")


if __name__ == "__main__":
    run_pipeline()
