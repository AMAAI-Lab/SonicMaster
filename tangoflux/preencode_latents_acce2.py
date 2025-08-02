import os
import glob
import torch
import torchaudio
from tqdm import tqdm
from diffusers import AutoencoderOobleck
from accelerate import Accelerator
from utils import pad_wav


def read_wav_file(filename, duration_sec):
    info = torchaudio.info(filename)
    sample_rate = info.sample_rate
    num_frames = int(sample_rate * duration_sec)

    waveform, sr = torchaudio.load(filename, num_frames=num_frames)

    # Resample
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
    waveform = resampler(waveform)

    # Convert mono to stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    # Pad each channel
    target_length = int(44100 * duration_sec)
    padded_left = pad_wav(waveform[0], target_length)
    padded_right = pad_wav(waveform[1], target_length)

    return torch.stack([padded_left, padded_right])


def main():
    accelerator = Accelerator()
    device = accelerator.device

    input_dir = "/data/shared/JamendoMastero/degrads2"
    output_dir = "/data/shared/JamendoMastero/degradspt2"
    duration_sec = 30
    batch_size = 8

    # Load VAE and prepare for multi-GPU
    vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="vae"
    )
    vae.eval()
    vae.requires_grad_(False)
    vae = accelerator.prepare(vae)

    # Partition file list by rank
    flac_files = sorted(glob.glob(os.path.join(input_dir, "*.flac")))
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    flac_files = flac_files[rank::world_size]  # Distribute files across processes

    batch_waveforms = []
    batch_filenames = []

    for flac_file in tqdm(flac_files, desc=f"Rank {rank} Encoding", disable=not accelerator.is_main_process):
        try:
            waveform = read_wav_file(flac_file, duration_sec)
            batch_waveforms.append(waveform)
            batch_filenames.append(os.path.basename(flac_file))

            if len(batch_waveforms) == batch_size or flac_file == flac_files[-1]:
                batch_tensor = torch.stack(batch_waveforms).to(device)

                with torch.no_grad():
                    latents = vae.encode(batch_tensor).latent_dist.mode()
                    latents = latents.transpose(1, 2)  # [B, T, C]

                for fname, latent in zip(batch_filenames, latents.cpu()):
                    outpath = os.path.join(output_dir, fname.replace(".flac", ".pt"))
                    torch.save(latent, outpath)

                batch_waveforms.clear()
                batch_filenames.clear()

        except Exception as e:
            print(f"Error processing {flac_file} on rank {rank}: {e}")


if __name__ == "__main__":
    main()
