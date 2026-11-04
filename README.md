# AnimeGen

[![Runpod](https://api.runpod.io/badge/ujwaltheja/AnimeGen)](https://console.runpod.io/hub/listing/ujwaltheja/AnimeGen)

Open-weight **image-to-video** on GPU — **HunyuanVideo 1.5**, **LTX-2.3 (video + audio)**, and **Wan 2.2** — locally or on **[Runpod](https://www.runpod.io)** Serverless.

## Model profiles

Set `--model <profile>` or `ANIMEGEN_MODEL`. Defaults: **`hunyuan15_i2v`**.

| Profile | Engine | Model | VRAM (typical) | Highlights |
|--------|--------|--------|----------------|------------|
| **`hunyuan15_i2v`** | Hunyuan 1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v` | 40 GB offload / 80 GB GPU | **Default** — cinematic 720p I2V |
| **`ltx23_i2v`** | LTX-2.3 | `diffusers/LTX-2.3-Diffusers` | 48 GB+ | **Native synchronized audio** + video |
| **`ltx23_distilled`** | LTX-2.3 | `diffusers/LTX-2.3-Distilled-Diffusers` | 40 GB+ | 8-step fast LTX with audio |
| **`hunyuan_i2v`** | HunyuanVideo | `hunyuanvideo-community/HunyuanVideo-I2V` | 24–48 GB | Original 2025 I2V checkpoint |
| **`i2v_a14b`** | Wan 2.2 MoE | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 48 GB offload | Strong motion, Apache 2.0 |
| **`ti2v_5b`** | Wan 2.2 | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 24 GB | Fast previews, Apache 2.0 |

Licenses differ (Tencent Hunyuan Community, LTX commercial threshold, Apache for Wan). Read each model card before production use.

## Project layout

```
config/models.yaml       # Profiles and defaults
src/animegen/engines/    # wan | hunyuan15 | hunyuan | ltx2
scripts/animate.py       # Local CLI
runpod/                  # Serverless worker + Dockerfile
```

## Local setup

```bash
cd AnimeGen
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cp .env.example .env
export PYTHONPATH=src

# Default Hunyuan 1.5 I2V
python scripts/animate.py photo.png "slow zoom, wind in hair" -o outputs/out.mp4

# LTX with generated audio
python scripts/animate.py photo.png "ocean waves, ambient surf" --model ltx23_i2v -o outputs/ltx.mp4

# Wan MoE (Apache 2.0)
python scripts/animate.py photo.png "anime idle motion" --model i2v_a14b -o outputs/wan.mp4
```

**Memory:** `ANIMEGEN_LOAD_MODE=offload` (default for large profiles) or `gpu` on 80 GB cards.

## Runpod Hub

This repo is configured for the [Runpod Hub](https://console.runpod.io/hub) (`.runpod/hub.json`, `.runpod/tests.json`, `.runpod/handler.py`, `.runpod/Dockerfile`). Create a **new GitHub release** after each Hub-related change — the Hub indexes **releases**, not commits.

Deploy from GitHub:

[![Runpod](https://api.runpod.io/badge/ujwaltheja/AnimeGen)](https://console.runpod.io/hub/listing/ujwaltheja/AnimeGen)

**Hub build passed but tests failed with “no instances available”?** That is Runpod GPU capacity, not your Docker image. Cut a **new release** to retry the scheduler, or edit `.runpod/tests.json` `gpuTypeId` (e.g. `NVIDIA RTX A5000`, `NVIDIA L4`, `NVIDIA A40`). To skip GPU validation temporarily, rename `.runpod/tests.json` → `.runpod/tests_.json` and release again.

### Serverless request (API)

```bash
docker build --platform linux/amd64 -t YOUR_USER/animegen-i2v .
docker push YOUR_USER/animegen-i2v:latest
```

1. **Serverless** → new endpoint → your image.
2. **GPU:** **40 GB+** for `hunyuan15_i2v` / `ltx23_distilled`; **48–80 GB** for `ltx23_i2v`.
3. Env: `HF_TOKEN`, `ANIMEGEN_MODEL=hunyuan15_i2v`, `ANIMEGEN_LOAD_MODE=offload`.
4. Attach a **network volume** at `/runpod-volume` for weight cache (`HF_HOME`).

### Request

```json
{
  "input": {
    "image_base64": "<base64>",
    "prompt": "Cinematic push-in, natural motion",
    "model": "ltx23_i2v",
    "num_frames": 121,
    "seed": 42
  }
}
```

Response: `video_base64`, `has_audio` (true for LTX profiles).

```bash
python scripts/runpod_client.py input.png "prompt" --model ltx23_i2v -o outputs/runpod.mp4
```

## Notes

- Requires **diffusers from GitHub** (see `requirements.txt`).
- **LTX:** width/height must be divisible by 32; frame count `8k+1`.
- **Hunyuan 1.5:** optional Flash attention via `pip install kernels` on supported GPUs.

## License

Application code: MIT. Weights: see Hugging Face model cards per profile.
