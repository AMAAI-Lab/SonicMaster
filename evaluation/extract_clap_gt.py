import os
import json
import numpy as np
from tqdm import tqdm
import soundfile as sf
import librosa
from laion_clap import CLAP_Module

def load_mono_audio(path, target_sr=44100):
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def read_unique_ids(jsonl_path):
    seen = set()
    unique_ids = []
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            clean_id = entry["original_id"] #if "original_id" in entry else entry["id"]
            if clean_id not in seen:
                seen.add(clean_id)
                unique_ids.append(clean_id)
    return unique_ids

def extract_embeddings(model, ids, audio_folder):
    embeddings = []
    filenames = []
    for audio_id in tqdm(ids, desc="Extracting clean embeddings"):
        audio_path = os.path.join(audio_folder, f"{audio_id}.flac")
        if not os.path.exists(audio_path):
            print(f"Warning: file {audio_path} not found, skipping.")
            continue
        emb = model.get_audio_embedding_from_filelist([audio_path], use_tensor=False)
        embeddings.append(emb[0])  # emb is a list of 1
        filenames.append(audio_id)
    return np.stack(embeddings), filenames


def save_embeddings(output_path, embeddings, filenames):
    np.savez(output_path, embeddings=embeddings, filenames=filenames)

if __name__ == "__main__":
    jsonl_file = "/mastering/testset.jsonl"
    clean_audio_folder = "/dataset/targets"
    output_npz = "/mastering/processedclap/clean_embeddings.npz"

    os.makedirs(os.path.dirname(output_npz),exist_ok=True)

    print("Loading CLAP model...")
    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    # model.load_audio_model()  # <- REQUIRED
    model.eval()


    print("Reading unique audio IDs...")
    unique_ids = read_unique_ids(jsonl_file)

    print(f"Found {len(unique_ids)} unique clean IDs.")

    embeddings, filenames = extract_embeddings(model, unique_ids, clean_audio_folder)
    save_embeddings(output_npz, embeddings, filenames)

    print(f"Saved {len(filenames)} clean embeddings to {output_npz}")
