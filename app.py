import os
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torchaudio
from einops import rearrange
import base64
import tempfile
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# Set Hugging Face cache location
os.environ["TRANSFORMERS_CACHE"] = "/app/hf_cache"

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device)
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

class AudioRequest(BaseModel):
    prompt: str
    duration: float = 10.0

@app.post("/generate")
async def generate_audio(request: AudioRequest):
    conditioning = [{
        "prompt": request.prompt,
        "seconds_start": 0,
        "seconds_total": request.duration
    }]

    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )

    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
    output = output.mul(32767).to(torch.int16).cpu()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        torchaudio.save(f.name, output, sample_rate)
        f.seek(0)
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return {
        "prompt": request.prompt,
        "duration": request.duration,
        "audio_base64": audio_b64
    }
