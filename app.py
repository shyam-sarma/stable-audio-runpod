from fastapi import FastAPI
from pydantic import BaseModel
from stable_audio_tools import generate_audio
import base64

app = FastAPI()

class AudioRequest(BaseModel):
    prompt: str
    duration: float = 5.0  # in seconds

@app.post("/generate")
async def generate_audio_endpoint(request: AudioRequest):
    audio_bytes = generate_audio(
        input_prompt=request.prompt,
        output_path=None,
        seed=42,
        duration=request.duration,
        cfg_scale=7.0,
        top_k=250,
        top_p=0.0,
        temperature=1.0,
        classifier_free_guidance=True
    )

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return {
        "prompt": request.prompt,
        "duration": request.duration,
        "audio_base64": audio_b64
    }
