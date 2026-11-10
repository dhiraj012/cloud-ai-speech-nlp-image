from faster_whisper import WhisperModel

def transcribe_audio(audio_path="audio.wav"):
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, language="en")

    text = " ".join(segment.text.strip() for segment in segments)

    if not text:
        raise ValueError("No speech detected.")

    return text

if __name__ == "__main__":
    result = transcribe_audio()
    print("Transcribed Text:", result)
