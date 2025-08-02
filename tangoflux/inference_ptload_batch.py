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
from model import TangoFlux
from datasets import load_dataset, Audio
from utils import Text2AudioDataset, read_wav_file, pad_wav

from diffusers import AutoencoderOobleck
import torchaudio
from safetensors.torch import load_file
import soundfile as sf


logger = get_logger(__name__)


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

    per_device_batch_size = int(config["training"]["per_device_batch_size"])

    output_dir = config["paths"]["output_dir"]

    jsonfile = config["paths"]["infer_file"]

    # accelerator = Accelerator(
    #     gradient_accumulation_steps=gradient_accumulation_steps,
    #     **accelerator_log_kwargs,
    # )


    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory creation and wandb tracking
    # if accelerator.is_main_process:
    #     if output_dir is None or output_dir == "":
    #         output_dir = "saved/" + str(int(time.time()))

    #         if not os.path.exists("saved"):
    #             os.makedirs("saved")

    #         os.makedirs(output_dir, exist_ok=True)

    #     elif output_dir is not None:
    #         os.makedirs(output_dir, exist_ok=True)

    #     os.makedirs("{}/{}".format(output_dir, "outputs"), exist_ok=True)
    #     with open("{}/summary.jsonl".format(output_dir), "a") as f:
    #         f.write(json.dumps(dict(vars(args))) + "\n\n")

    #     accelerator.project_configuration.automatic_checkpoint_naming = False

    #     wandb.init(
    #         project="Text to Audio Flow matching",
    #         settings=wandb.Settings(_disable_stats=True),
    #     )

    # accelerator.wait_for_everyone()

    # Get the datasets
    data_files = {}

    if config["paths"]["infer_file"] != "":
        data_files["infer"] = config["paths"]["infer_file"]

    extension = "json"
    raw_datasets = load_dataset(extension, data_files=data_files)
    text_column, alt_text_column, audio_column, deg_audio_column = args.text_column, args.alt_text_column, args.audio_column, args.deg_audio_column

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

    ## Freeze vae
    # for param in vae.parameters():
    #     vae.requires_grad = False
    #     vae.eval()

    ## Freeze text encoder param
    for param in model.text_encoder.parameters():
        param.requires_grad = False
        model.text_encoder.eval()

    prefix = args.prefix

    # with accelerator.main_process_first():
    #     infer_dataset = Text2AudioDataset(
    #         raw_datasets["infer"],
    #         prefix,
    #         text_column,
    #         audio_column,
    #         deg_audio_column,
    #         "duration",
    #         args.num_examples,
    #     )
    #     accelerator.print(
    #         "Num instances in train: {}, validation: {}, test: {}".format(
    #             train_dataset.get_num_instances(),
    #             eval_dataset.get_num_instances(),
    #             test_dataset.get_num_instances(),
    #         )
    #     )

    fs=44100
    filenames=[]
    with open(jsonfile, "r", encoding="utf-8") as infile:
        for line in infile:
            a=json.loads(line)
            filenames.append(os.path.basename(a["location"]))

    infer_dataset = Text2AudioDataset(
        raw_datasets["infer"],
        prefix,
        text_column,
        alt_text_column,
        audio_column,
        deg_audio_column,
        "duration",
        args.num_examples,
    )

    infer_dataloader = DataLoader(
        infer_dataset,
        shuffle=False,
        # batch_size=config["training"]["per_device_batch_size"],
        batch_size=8,
        collate_fn=infer_dataset.collate_fn,
    )



    total_batch_size = per_device_batch_size

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(math.ceil(len(infer_dataloader) / total_batch_size))
    )


    infer_outputs=[]
    # wave_list=[]
    model.eval()
    i=0

    inference_output_dir = os.path.join(output_dir, "inference_seed27full10sec_nocond40g1_ek100")
    os.makedirs(inference_output_dir, exist_ok=True)
    for step, batch in enumerate(infer_dataloader):
        # inference_batch = next(iter(infer_dataloader))
        # with accelerator.accumulate(model) and torch.no_grad():
        with torch.no_grad():

            text, alt_text, audios, deg_audios, duration, _ = batch

            audio_list = []
            deg_audio_list = []
            for audio_path,deg_audio_path in zip(audios,deg_audios):

                loaded_tensor=torch.load(audio_path)
                audio_list.append(loaded_tensor)

                loaded_tensor=torch.load(deg_audio_path)
                deg_audio_list.append(loaded_tensor)

            audio_latent = torch.stack(audio_list, dim=0)
            audio_latent = audio_latent.to(device)
            deg_audio_latent = torch.stack(deg_audio_list, dim=0)
            deg_audio_latent = deg_audio_latent.to(device)

            text = [""]*len(text)

            inferred_result = model.inference_flow(
                deg_audio_latent,
                text,
                # audiocond_latents=audio_latent,
                audiocond_latents=None,
                num_inference_steps=100,
                timesteps=None,
                guidance_scale=1,
                duration=duration,
                seed=0,
                disable_progress=False,
                num_samples_per_prompt=1,
                callback_on_step_end=None,
                solver="Euler", #Euler or rk4
            )
            infer_outputs.append(inferred_result)
            wave_list=[]

            wave = vae.decode(inferred_result.transpose(2, 1)).sample.cpu()
            wave_list.append(wave)

            for k in range(len(wave_list[0])):
                sf.write(os.path.join(inference_output_dir,filenames[i].replace(".pt", ".flac")), wave_list[0][k].numpy().T, samplerate=fs, format='FLAC')
                i+=1

    # if accelerator.is_main_process:


    # for i, out in enumerate(infer_outputs):
    #     torch.save(out.cpu(), os.path.join(inference_output_dir, f"sample_{i}.pt"))
    # for i, out in enumerate(wave_list):
    #     sf.write(os.path.join(inference_output_dir,f"sample_{i}.flac"), out.numpy().T, samplerate=fs, format='FLAC')


        # torch.save(out.cpu(), os.path.join(inference_output_dir, f"sample_{i}.wav"))
                # if accelerator.sync_gradients:
                #     progress_bar.update(1)
                #     completed_steps += 1




if __name__ == "__main__":
    main()
