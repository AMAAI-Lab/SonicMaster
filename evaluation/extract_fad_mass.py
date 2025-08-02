import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from laion_clap import CLAP_Module
from scipy.linalg import sqrtm

def load_clean_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    emb = data["embeddings"]
    fnames = data["filenames"]
    fname_to_emb = {fname: emb[i] for i, fname in enumerate(fnames)}
    return fname_to_emb

# def load_test_entries(jsonl_path):
#     entries = []
#     with open(jsonl_path, "r") as f:
#         for line in f:
#             entry = json.loads(line)
#             entries.append((entry["id"], entry["original_id"]))
#     return entries


def load_test_entries(jsonl_path):
    entries = []
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            degradations = entry.get("degradations", [])
            # if len(degradations) > 1:
            #     continue
            # if len(degradations) < 2:
            #     continue
            entries.append((entry["id"], entry["original_id"]))
    return entries


def extract_degraded_embeddings(model, entries, degraded_folder, clean_lookup):
    degraded_embeddings = []
    clean_embeddings = []

    for degraded_id, original_id in tqdm(entries, desc=f"Extracting from {os.path.basename(degraded_folder)}"):
        degraded_path = os.path.join(degraded_folder, f"{degraded_id}.flac")
        if not os.path.exists(degraded_path):
            print(f"⚠️  Missing degraded file: {degraded_path}")
            continue
        if original_id not in clean_lookup:
            print(f"⚠️  Clean embedding for '{original_id}' not found.")
            continue
        try:
            emb = model.get_audio_embedding_from_filelist([degraded_path], use_tensor=False)
            if len(emb) == 0:
                print(f"❌ No embedding for {degraded_path}")
                continue
            degraded_embeddings.append(emb[0])
            clean_embeddings.append(clean_lookup[original_id])
        except Exception as e:
            print(f"❌ Failed on {degraded_path}: {e}")
            continue

    return np.stack(clean_embeddings), np.stack(degraded_embeddings)

def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

def compute_fad(clean_embs, degraded_embs):
    mu1 = np.mean(clean_embs, axis=0)
    mu2 = np.mean(degraded_embs, axis=0)
    sigma1 = np.cov(clean_embs, rowvar=False)
    sigma2 = np.cov(degraded_embs, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)

if __name__ == "__main__":
    # Inputs
    jsonl_file = "/testset_pt.jsonl"
    clean_npz = "/mastering/processedclap/clean_embeddings.npz"
    folders = [
        "outputs/run1",
        "outputs/run2"
    ]
    output_csv = "/evaluationfinal/fad_results_all.csv"

    print("📦 Loading clean embeddings...")
    clean_lookup = load_clean_embeddings(clean_npz)

    print("📑 Loading testset entries...")
    entries = load_test_entries(jsonl_file)

    print("🎧 Loading CLAP model...")
    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    model.eval()

    results = []

    for folder in folders:
        print(f"\n🚀 Processing: {folder}")
        clean_embs, degraded_embs = extract_degraded_embeddings(model, entries, folder, clean_lookup)

        if len(clean_embs) == 0 or len(degraded_embs) == 0:
            print(f"❌ Skipping {folder} — no valid embeddings")
            continue

        print(f"📊 Computing FAD for {len(clean_embs)} pairs...")
        fad_value = compute_fad(clean_embs, degraded_embs)

        results.append({
            "folder": os.path.basename(folder),
            "num_pairs": len(clean_embs),
            "fad": fad_value
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print("\n✅ All FAD results saved:")
    print(df)
