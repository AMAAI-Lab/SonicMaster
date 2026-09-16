# inference_fullsong.py
import argparse
import os
from pathlib import Path
from time import time

import torch
import torchaudio
import soundfile as sf
import yaml
from safetensors.torch import load_file
from diffusers import AutoencoderOobleck

# Local imports (repo root is on sys.path when this file is executed)
from fullsong_chunking import (
    CARRY_LATENT_FRAMES,
    MAIN_LATENT_FRAMES,
    SONICMASTER_SAMPLE_RATE,
    TRAINED_DURATION_SECONDS,
    crossfade_and_trim,
    make_overlapping_chunks,
    make_vae_native_geometry,
)
from model import TangoFlux

hf_token = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

def parse_args():
    p = argparse.ArgumentParser("Single-sample inference for SonicMaster")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to model.safetensors (or directory containing it).")
    p.add_argument("--input", type=str, required=True,
                   help="Path to degraded input audio (wav/flac/etc).")
    p.add_argument("--prompt", type=str, required=True,
                   help="Text prompt guiding the enhancement/restoration.")
    p.add_argument("--output", type=str, required=True,
                   help="Output audio path (use .wav or .flac extension).")

    # Optional knobs (safe defaults)
    p.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "tangoflux_config.yaml"),
                   help="YAML config defining model sizes/hparams.")
    p.add_argument("--vae_batch_size", type=int, default=10, help="Batch size for VAE encoding over chunks.")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--solver", type=str, default="Euler")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    t0 = time()
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --------- Resolve checkpoint path (file or directory) ----------
    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir():
        candidate = ckpt_path / "model.safetensors"
        if not candidate.exists():
            raise FileNotFoundError(f"Could not find model.safetensors in {ckpt_path}")
        ckpt_path = candidate
    elif not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # --------- Load config & model ----------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model = TangoFlux(config=cfg["model"])

    weights = load_file(str(ckpt_path))
    model.load_state_dict(weights, strict=False)
    model.to(device).eval()

    # Freeze text encoder params
    for p in model.text_encoder.parameters():
        p.requires_grad = False
    model.text_encoder.eval()

    # --------- Load VAE ----------
    vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="vae",use_auth_token=hf_token,
    ).to(device)
    vae.eval()

    geometry = make_vae_native_geometry(
        vae_hop_length=getattr(vae, "hop_length", 0),
        model_audio_seq_len=model.audio_seq_len,
        vae_sample_rate=getattr(vae, "sampling_rate", 0),
    )

    # --------- Read & standardize input ----------
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input audio not found: {in_path}")

    audio, sr = torchaudio.load(str(in_path))  # [C, T]
    # Force stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]

    # The published VAE and SonicMaster checkpoint are trained at 44.1 kHz.
    if sr != SONICMASTER_SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            audio, sr, SONICMASTER_SAMPLE_RATE
        )
        sr = SONICMASTER_SAMPLE_RATE

    audio = audio.to(device)

    # --------- Chunking ----------
    fs = SONICMASTER_SAMPLE_RATE
    T = audio.shape[1]
    chunks = make_overlapping_chunks(audio, geometry)

    if not chunks:
        raise RuntimeError("No audio content to process.")

    # --------- Pre-encode degraded chunks with VAE (batched) ----------
    chunk_tensor = torch.stack(chunks)  # [N, 2, T]
    latents = []
    for b in range(0, chunk_tensor.shape[0], args.vae_batch_size):
        batch = chunk_tensor[b:b + args.vae_batch_size].to(device)
        if batch.shape[-1] != geometry.chunk_size:
            raise RuntimeError(
                "Chunk batch has "
                f"{batch.shape[-1]} samples; expected {geometry.chunk_size}."
            )
        z = vae.encode(batch).latent_dist.mode()  # [B, C, T']
        if z.shape[-1] != MAIN_LATENT_FRAMES:
            raise RuntimeError(
                "VAE main encode length mismatch: "
                f"{batch.shape[-1]} samples produced {z.shape[-1]} frames; "
                f"expected {MAIN_LATENT_FRAMES}."
            )
        latents.append(z)
    degraded_latents = torch.cat(latents, dim=0)  # [N, C, T']

    # --------- Inference loop with conditional carry ----------
    decoded_chunks = []
    prev_cond = None
    g = torch.Generator(device=device).manual_seed(args.seed)

    for i in range(degraded_latents.shape[0]):
        # model expects [1, T', C] (transpose from [C, T'] if needed by your impl)
        z_in = degraded_latents[i].unsqueeze(0).transpose(1, 2)  # [1, T', C]

        result_latent = model.inference_flow(
            z_in,
            args.prompt,
            audiocond_latents=prev_cond,        # None for first chunk
            num_inference_steps=args.num_inference_steps,
            timesteps=None,
            guidance_scale=args.guidance_scale,
            duration=TRAINED_DURATION_SECONDS,
            seed=args.seed,
            disable_progress=True,
            num_samples_per_prompt=1,
            callback_on_step_end=None,
            solver=args.solver,
        )

        # Decode to waveform on CPU for stitching
        wav = vae.decode(result_latent.transpose(2, 1)).sample.cpu()  # [1, 2, T]
        if wav.shape[-1] != geometry.chunk_size:
            raise RuntimeError(
                "VAE decoded chunk length mismatch; refusing to stitch: "
                f"chunk {i} produced {wav.shape[-1]} samples, "
                f"expected {geometry.chunk_size}."
            )
        # Safety clamp to [-1, 1]
        wav = torch.clamp(wav, -1.0, 1.0)
        decoded_chunks.append(wav)

        # Carry last overlap as conditioning (back on device)
        last = wav[:, :, -geometry.overlap:].to(device)
        if last.shape[-1] != geometry.overlap:
            raise RuntimeError(
                "Conditioning carry has "
                f"{last.shape[-1]} samples; expected {geometry.overlap}."
            )
        carry_latent = vae.encode(last).latent_dist.mode()
        if carry_latent.shape[-1] != CARRY_LATENT_FRAMES:
            raise RuntimeError(
                "VAE carry encode length mismatch: "
                f"{last.shape[-1]} samples produced {carry_latent.shape[-1]} frames; "
                f"expected {CARRY_LATENT_FRAMES}."
            )
        prev_cond = carry_latent.transpose(1, 2)  # [1, T', C]

    # --------- Crossfade stitch ----------
    # The last partial VAE-native chunk is right-padded for inference. Remove
    # only its excess stitched tail by preserving the exact resampled length.
    final = crossfade_and_trim(decoded_chunks, geometry, T)

    # --------- Save (honor extension) ----------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = final.squeeze(0).numpy().T  # [T, 2]

    ext = out_path.suffix.lower()
    if ext == ".wav":
        sf.write(out_path.as_posix(), data, fs, format="WAV")
    elif ext == ".flac":
        sf.write(out_path.as_posix(), data, fs, format="FLAC")
    else:
        # Default to WAV if unknown extension
        sf.write(out_path.as_posix(), data, fs, format="WAV")

    print(f"Saved: {out_path}")
    print(f"Elapsed: {time() - t0:.2f}s")


if __name__ == "__main__":
    main()
