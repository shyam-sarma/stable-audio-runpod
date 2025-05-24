from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor, StableAudioOpenModel
import torch
import torchaudio
import tempfile
import base64

app = FastAPI()

class AudioRequest(BaseModel):
    prompt: str
    duration: float = 5.0  # in seconds

processor = AutoProcessor.from_pretrained("stabilityai/stable-audio-open-1.0")
model = StableAudioOpenModel.from_pretrained("stabilityai/stable-audio-open-1.0")

@app.post("/generate")
async def generate_audio(request: AudioRequest):
    inputs = processor(
        text=[request.prompt],
        padding=True,
        return_tensors="pt",
        sampling_rate=16000,
        duration=request.duration,
    )

    with torch.no_grad():
        output = model(**inputs).audio_values[0]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        torchaudio.save(f.name, output.unsqueeze(0), 16000)
        f.seek(0)
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "prompt": request.prompt,
        "duration": request.duration,
        "audio_base64": audio_b64
    }
