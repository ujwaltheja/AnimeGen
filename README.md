# AnimeGen

Open-weight **image-to-video** animation using **[Wan 2.2](https://huggingface.co/Wan-AI)** on GPU — locally or on **[Runpod](https://www.runpod.io)** Serverless.

## Why Wan 2.2?

| Profile | Hugging Face model | VRAM | License | Best for |
|--------|---------------------|------|---------|----------|
| **`ti2v_5b`** (default) | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | ~24 GB | Apache 2.0 | Runpod 24 GB GPUs (RTX 4090, A5000), fast iteration |
| **`i2v_a14b`** | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | ~48 GB+ | Apache 2.0 | Highest open I2V quality on A6000 / A100 |

Other open models (CogVideoX-5B, HunyuanVideo, LTX-2.3) are listed in community roundups; Wan 2.2 is the default here because of **Apache 2.0**, strong I2V quality, Diffusers support, and official Runpod endpoint docs for Wan I2V LoRA workflows.

## Project layout

```
config/models.yaml     # Model profiles and defaults
src/animegen/          # Inference engine
scripts/animate.py     # Local GPU CLI
runpod/                # Serverless handler + Dockerfile
```

## Local setup (GPU)

1. Create a virtualenv and install PyTorch for your CUDA version, then dependencies:

```bash
cd AnimeGen
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .  # optional; or export PYTHONPATH=src
```

2. Copy env template and set `HF_TOKEN` if needed:

```bash
cp .env.example .env
```

3. Run animation:

```bash
export PYTHONPATH=src
python scripts/animate.py path/to/image.png "Anime character smiles and hair blows in the wind" -o outputs/out.mp4
```

Use `--model i2v_a14b` on a 48 GB+ GPU for the MoE I2V checkpoint.

## Runpod Serverless

### Build and push worker image

```bash
docker build --platform linux/amd64 -f runpod/Dockerfile -t YOUR_USER/animegen-wan-i2v .
docker push YOUR_USER/animegen-wan-i2v:latest
```

### Create endpoint

1. Runpod console → **Serverless** → **New endpoint** → import your Docker image.
2. **GPU**: 24 GB for `ti2v_5b`, **48 GB+** for `i2v_a14b`.
3. Set env: `HF_TOKEN`, optional `ANIMEGEN_MODEL=ti2v_5b`.
4. Attach a **network volume** (recommended) so `HF_HOME` caches weights across cold starts.

### Request shape

```json
{
  "input": {
    "image_base64": "<base64>",
    "prompt": "Smooth parallax, subtle motion, maintain character identity",
    "model": "ti2v_5b",
    "num_frames": 81,
    "num_inference_steps": 40,
    "seed": 42
  }
}
```

Response includes `video_base64` (MP4).

### Test from your machine

```bash
cp .env.example .env
# RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...
python scripts/runpod_client.py input.png "your prompt" -o outputs/runpod.mp4
```

### Local handler test

```bash
export PYTHONPATH=src
cd runpod && python handler.py
```

Uses `runpod/test_input.json` when run locally.

## Runpod GPU Pod (interactive)

For long sessions or ComfyUI, rent a **GPU Pod** (24 GB+) instead of Serverless, clone this repo, install requirements, and run `scripts/animate.py`. Same model IDs as above.

## Notes

- **Diffusers**: Wan 2.2 needs recent `diffusers` from GitHub (pinned in `requirements.txt`).
- **First run** downloads multi-GB weights from Hugging Face; use a volume on Runpod.
- **Negative prompt** defaults match Wan’s recommended Chinese quality guardrails (see `config/models.yaml`).

## License

Code in this repo: MIT. Model weights follow each checkpoint’s license on Hugging Face (Wan 2.2: Apache 2.0).
