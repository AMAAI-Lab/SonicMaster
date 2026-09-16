import time
import argparse
import json
import logging
import math
import os
import yaml
from pathlib import Path
import diffusers
import datasets
import numpy as np
import pandas as pd
import transformers
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
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
from datasets import load_dataset, Audio
from utils import Text2AudioDataset, read_wav_file, pad_wav

from diffusers import AutoencoderOobleck
import torchaudio
from safetensors.torch import load_file
import soundfile as sf
from time import time

logger = get_logger(__name__)

timestart=time()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rectified flow for text to audio generation task."
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=-1,
        help="How many examples to use for training and validation.",
    )

    parser.add_argument(
        "--text_column",
        type=str,
        default="prompt",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--alt_text_column",
        type=str,
        default="alt_prompt",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="original_location",
        help="The name of the column in the datasets containing the target audio paths.",
    )
    parser.add_argument(
        "--deg_audio_column",
        type=str,
        default="location",
        help="The name of the column in the datasets containing the degraded audio paths.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/tangoflux_config.yaml",
        help="Config file defining the model size as well as other hyper parameter.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Add prefix in text prompts.",
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="best",
        help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
    )

    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="/outputs/seed27full10sec/epoch_40",
        help="Path to the model checkpoint.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # accelerator_log_kwargs = {}
    device="cuda" if torch.cuda.is_available() else "cpu"
    def load_config(config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    config = load_config(args.config)

    # per_device_batch_size = int(config["training"]["per_device_batch_size"])

    # output_dir = config["paths"]["output_dir"]

    # jsonfile = config["paths"]["infer_file"]

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets
    # data_files = {}

    # if config["paths"]["infer_file"] != "":
    #     data_files["infer"] = config["paths"]["infer_file"]

    # extension = "json"
    # raw_datasets = load_dataset(extension, data_files=data_files)
    # text_column, alt_text_column, audio_column, deg_audio_column = args.text_column, args.alt_text_column, args.audio_column, args.deg_audio_column

    model = TangoFlux(config=config["model"])
    # model.load_state_dict(torch.load(os.path.join(args.model_ckpt,"model_1.safetensors")))

    weights = load_file(os.path.join(args.model_ckpt,"model.safetensors"))
    model.load_state_dict(weights, strict=False)
    model.to(device)
    model.eval()

    vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="vae"
    )
    vae.to(device)
    vae.eval()

    geometry = make_vae_native_geometry(
        vae_hop_length=getattr(vae, "hop_length", 0),
        model_audio_seq_len=model.audio_seq_len,
        vae_sample_rate=getattr(vae, "sampling_rate", 0),
    )


    ## Freeze text encoder param
    for param in model.text_encoder.parameters():
        param.requires_grad = False
        model.text_encoder.eval()

    # prefix = args.prefix






    jsonl_path = "/fullsongs/500_full_deg_short.jsonl"
    # input_dir = "/path/to/flacs"
    output_dir = "/outputs/fullsongs/full10sec40g1"
    os.makedirs(output_dir, exist_ok=True)

    fs = SONICMASTER_SAMPLE_RATE

    # Load JSONL
    with open(jsonl_path, "r") as f:
        entries = [json.loads(line.strip()) for line in f]

    for entry in tqdm(entries):
        song_id = entry["id"]
        # input_path = os.path.join(input_dir, f"{song_id}.flac")
        input_path = entry["location"]
        text = entry["prompt"]

        if not os.path.exists(input_path):
            print(f"Missing file: {input_path}")
            continue

        # Load stereo audio
        audio, sr = torchaudio.load(input_path)
        if sr != fs:
            raise ValueError(f"Expected {fs} Hz, got {sr}")
        audio = audio.to(device)  # [2, T]

        # Chunking degraded audio
        target_length = audio.shape[1]
        chunks = make_overlapping_chunks(audio, geometry)
        if not chunks:
            raise RuntimeError(f"No audio content to process: {input_path}")

        # Pre-encode degraded chunks
        # with torch.no_grad():
        #     # degraded_latents = vae.encode(torch.stack(chunks)).latent  # [N, C, T']
        #     degraded_latents = vae.encode(torch.stack(chunks)).latent_dist.mode()  # [N, C, T']

        with torch.no_grad():
            degraded_latents_list = []
            batch_size = 10
            chunk_tensor = torch.stack(chunks).to(device)  # [N, 2, chunk_size]
            num_chunks = len(chunk_tensor)

            for b in range(0, num_chunks, batch_size):
                batch = chunk_tensor[b:b+batch_size]  # [B, 2, T]
                if batch.shape[-1] != geometry.chunk_size:
                    raise RuntimeError(
                        "Chunk batch has "
                        f"{batch.shape[-1]} samples; expected {geometry.chunk_size}."
                    )
                latent = vae.encode(batch).latent_dist.mode()  # [B, C, T']
                if latent.shape[-1] != MAIN_LATENT_FRAMES:
                    raise RuntimeError(
                        "VAE main encode length mismatch: "
                        f"{batch.shape[-1]} samples produced {latent.shape[-1]} "
                        f"frames; expected {MAIN_LATENT_FRAMES}."
                    )
                degraded_latents_list.append(latent)

            degraded_latents = torch.cat(degraded_latents_list, dim=0)  # [N, C, T']

        decoded_chunks = []
        prev_cond_latent = None



        for i in range(len(degraded_latents)):
            degraded_latent = degraded_latents[i].unsqueeze(0).transpose(1,2)  # [1, C, T]

            # Run model with optional conditioning
            with torch.no_grad():
                result_latent = model.inference_flow(
                    degraded_latent,
                    text,
                    audiocond_latents=prev_cond_latent,  # None for first chunk
                    num_inference_steps=10,
                    timesteps=None,
                    guidance_scale=1,
                    duration=TRAINED_DURATION_SECONDS,
                    seed=0,
                    disable_progress=False,
                    num_samples_per_prompt=1,
                    callback_on_step_end=None,
                    solver="Euler",
                )

                # Decode latent to waveform
                decoded_wave = vae.decode(result_latent.transpose(2, 1)).sample.cpu()  # [1, 2, T]
                if decoded_wave.shape[-1] != geometry.chunk_size:
                    raise RuntimeError(
                        "VAE decoded chunk length mismatch; refusing to stitch: "
                        f"chunk {i} produced {decoded_wave.shape[-1]} samples, "
                        f"expected {geometry.chunk_size}."
                    )
                decoded_chunks.append(decoded_wave)

                # Re-encode the VAE-native carry as conditioning for the next chunk.
                carry = decoded_wave[:, :, -geometry.overlap:].to(device)
                if carry.shape[-1] != geometry.overlap:
                    raise RuntimeError(
                        "Conditioning carry has "
                        f"{carry.shape[-1]} samples; expected {geometry.overlap}."
                    )
                carry_latent = vae.encode(carry).latent_dist.mode()
                if carry_latent.shape[-1] != CARRY_LATENT_FRAMES:
                    raise RuntimeError(
                        "VAE carry encode length mismatch: "
                        f"{carry.shape[-1]} samples produced "
                        f"{carry_latent.shape[-1]} frames; "
                        f"expected {CARRY_LATENT_FRAMES}."
                    )
                prev_cond_latent = carry_latent.transpose(1,2)  # [1, C, T'] ->[1, T', C]

        # Remove only the padded stitched tail; underflow is an explicit error.
        final_output = crossfade_and_trim(
            decoded_chunks, geometry, target_length
        )

        # Save to file
        output_path = os.path.join(output_dir, f"{song_id}_reconstructed.flac")
        sf.write(output_path, final_output.squeeze().numpy().T, samplerate=fs, format='FLAC')  # [T, 2]



    print(time()-timestart)



if __name__ == "__main__":
    main()
