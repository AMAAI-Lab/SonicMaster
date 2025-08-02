import os
import json
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm

REVERB_TAGS = {"small", "big", "real", "mix"}

def is_reverb_degradation(entry):
    # Only process if exactly one degradation, and that degradation is a reverb type
    degs = entry.get("degradations", [])
    return len(degs) == 1 and degs[0] in REVERB_TAGS

def load_audio(path, sr=44100):
    audio, file_sr = sf.read(path)
    if file_sr != sr:
        print(f"Warning: file {path} sample rate {file_sr} != expected {sr}")
    return audio, file_sr

def hpss_dereverb(audio, harmonic_attenuation_db=12.0):
    attenuation = 10 ** (-harmonic_attenuation_db / 20)
    
    if audio.ndim == 1:
        # Mono
        S = librosa.stft(audio, n_fft=2048)
        H, P = librosa.decompose.hpss(S)
        H_filtered = H * attenuation
        S_filtered = H_filtered + P
        dereverb_audio = librosa.istft(S_filtered)
    else:
        # Stereo or multichannel
        dereverb_audio = np.zeros_like(audio)
        n_channels = audio.shape[1]
        for ch in range(n_channels):
            channel_data = audio[:, ch]
            S = librosa.stft(channel_data, n_fft=2048)
            H, P = librosa.decompose.hpss(S)
            H_filtered = H * attenuation
            S_filtered = H_filtered + P
            channel_dereverb = librosa.istft(S_filtered)
            if len(channel_dereverb) == 0:
                print(f"Warning: channel {ch} produced empty output, using original audio")
                channel_dereverb = channel_data
            if len(channel_dereverb) < len(channel_data):
                channel_dereverb = np.pad(channel_dereverb, (0, len(channel_data) - len(channel_dereverb)))
            elif len(channel_dereverb) > len(channel_data):
                channel_dereverb = channel_dereverb[:len(channel_data)]
            dereverb_audio[:, ch] = channel_dereverb
    return dereverb_audio


def process_jsonl(jsonl_path, input_dir, output_dir, sr=44100):
    os.makedirs(output_dir, exist_ok=True)
    with open(jsonl_path, "r") as f:
        entries = [json.loads(line) for line in f]

    filtered = [e for e in entries if is_reverb_degradation(e)]

    print(f"Processing {len(filtered)} reverb-degraded files...")

    for entry in tqdm(filtered):
        audio_id = entry["id"]
        audio_path = os.path.join(input_dir, f"{audio_id}.flac")
        output_path = os.path.join(output_dir, f"{audio_id}_hpss.flac")

        if not os.path.exists(audio_path):
            print(f"Warning: Missing file {audio_path}, skipping.")
            continue

        try:
            audio, _ = load_audio(audio_path, sr)
            dereverb_audio = hpss_dereverb(audio)
            # Save with shape (samples, channels)
            if dereverb_audio.ndim == 2:
                sf.write(output_path, dereverb_audio, sr, format='FLAC')
            else:
                sf.write(output_path, dereverb_audio, sr, format='FLAC')
        except Exception as e:
            print(f"Failed to process {audio_id}: {e}")

if __name__ == "__main__":
    jsonl_path = "/testset_pt.jsonl"
    input_audio_folder = "/dataset/degrads2"
    output_audio_folder = "/mastering/dereverbed_with_hpss12db"
    os.makedirs(output_audio_folder,exist_ok=True)

    process_jsonl(jsonl_path, input_audio_folder, output_audio_folder)
