from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import base64
import tempfile

app = FastAPI()

class AudioRequest(BaseModel):
    prompt: str
    duration: int = 10  # seconds

generator = pipeline("text-to-audio", model="stabilityai/stable-audio-open-1.0")

@app.post("/generate")
async def generate_audio(request: AudioRequest):
    result = generator(request.prompt, max_new_tokens=request.duration * 50)
    audio_bytes = result["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        temp_path = f.name

    with open(temp_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "prompt": request.prompt,
        "duration": request.duration,
        "sample_rate": result["sampling_rate"],
        "audio_base64": audio_b64
    }
