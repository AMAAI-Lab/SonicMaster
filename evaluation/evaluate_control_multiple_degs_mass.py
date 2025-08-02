
import pyroomacoustics as pra

import json
import os
from tqdm import tqdm
import soundfile as sf
import numpy as np
import librosa
from collections import defaultdict
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import stft

import csv
import pandas as pd


metrics_by_degradation = defaultdict(list)


# === CONFIG ===
SR = 44100
fs = 44100
sr = 44100
DEGRADATION_GROUPS = {
    "EQ": ["xband", "mic", "bright", "dark", "airy", "boom", "clarity", "mud", "warm", "vocal"],
    "Dynamics": ["comp", "punch"],
    "Reverb": ["small", "big", "mix", "real"],
    "Amplitude": ["clip", "volume"],
    "Stereo": ["stereo"]
}



def save_summary_to_csv(summary, summary_multi, out_path="resultsummary.csv"):
    # Merge both dictionaries into rows
    all_rows = []

    # Regular metrics
    for degradation, stats in summary.items():
        row = {"degradation": degradation}
        row.update(stats)
        all_rows.append(row)

    # Multi-metrics
    for degradation, stats in summary_multi.items():
        row = {"degradation": degradation}
        row.update(stats)
        all_rows.append(row)

    # Get all unique column names
    all_keys = set()
    for row in all_rows:
        all_keys.update(row.keys())
    fieldnames = ["degradation"] + sorted(k for k in all_keys if k != "degradation")

    # Write to CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"✅ Summary saved to: {out_path}")


def modulation_spectrum_distance(x1, x2, fs=44100, n_fft=1024, hop_length=512, n_mod_bins=20):
    """
    Modulation Spectrum Distance for detecting excess reverb between two signals.
    """
    def get_modulation_spectrum(x):
        f, t, Zxx = stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length)
        mag = np.abs(Zxx)

        mod_spec = []
        for band in mag:
            envelope = band - np.mean(band)
            spectrum = np.abs(np.fft.fft(envelope))[:n_mod_bins]
            mod_spec.append(spectrum)

        mod_spec = np.array(mod_spec)
        mod_spec /= np.linalg.norm(mod_spec) + 1e-10
        return mod_spec.flatten()

    mod1 = get_modulation_spectrum(x1)
    mod2 = get_modulation_spectrum(x2)

    return euclidean(mod1, mod2)

def rms_energy(signal):
    return np.sqrt(np.mean(signal**2))

def spectral_flatness(y, sr):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2
    flatness = librosa.feature.spectral_flatness(S=S)
    return np.mean(flatness)


def stereo_energy_ratio(y_stereo):
    if y_stereo.ndim != 2 or y_stereo.shape[1] != 2:
        return 0.0  # mono or malformed input
    
    L = y_stereo[:, 0]
    R = y_stereo[:, 1]
    M = (L + R) / 2
    S = (L - R) / 2

    rms_M = np.sqrt(np.mean(M ** 2))
    rms_S = np.sqrt(np.mean(S ** 2))

    return rms_S / (rms_M + 1e-10)




def estimate_rt60(audio, sr=44100, energy_threshold_db=60):
    """
    Estimate RT60 using Schroeder integration of the energy decay curve.
    Returns RT60 in seconds or None if not estimable.
    """
    # Square the signal to get energy envelope
    energy = audio ** 2

    # Integrate backwards (Schroeder)
    decay_curve = np.cumsum(energy[::-1])[::-1]

    # Convert to dB (add small epsilon to avoid log(0))
    decay_db = 10.0 * np.log10(decay_curve + 1e-10)
    decay_db -= np.max(decay_db)  # normalize peak to 0 dB

    # Find where decay crosses -5 dB and -35 dB (or adjust window)
    try:
        t = np.arange(len(decay_db)) / sr
        start_idx = np.where(decay_db <= -5)[0][0]
        end_idx   = np.where(decay_db <= -35)[0][0]
    except IndexError:
        return None  # insufficient decay in the signal

    # Fit a line to the decay slope
    x = t[start_idx:end_idx]
    y = decay_db[start_idx:end_idx]
    slope, intercept = np.polyfit(x, y, 1)

    if slope >= 0:
        return None  # not decaying

    rt60 = -60.0 / slope
    return rt60

# def estimate_rt60(audio, sr):
#     try:
#         rt60 = pra.experimental.rt60.schroeder(audio, fs=sr)
#         return rt60
#     except Exception as e:
#         print(f"RT60 estimation failed: {e}")
#         return None


def dynamic_range_std(audio, frame_length=2048, hop_length=1024):
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    rms = np.sqrt(np.mean(frames**2, axis=0))
    return np.std(rms)


def transient_strength(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    return np.mean(onset_env)


def multiband_spectral_profile(y, sr, bands):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    profile = []

    for low, high in bands:
        band = np.logical_and(freqs >= low, freqs < high)
        energy = np.sum(S[band, :])
        profile.append(energy)

    profile = np.array(profile)
    return profile / (np.sum(profile) + 1e-10)  # Normalize to unit sum


def spectral_balance_metrics(clean, degraded, output, sr=44100):
    bands = [
        (20, 60), (60, 250), (250, 500), (500, 2000),
        (2000, 4000), (4000, 6000), (6000, 10000),
        (10000, 16000), (16000, 20000)
    ]

    p_clean = multiband_spectral_profile(clean, sr, bands)
    p_degr = multiband_spectral_profile(degraded, sr, bands)
    p_out  = multiband_spectral_profile(output, sr, bands)

    # Compute distance metrics
    cos_degr = cosine(p_clean, p_degr)
    cos_out  = cosine(p_clean, p_out)
    euc_degr = euclidean(p_clean, p_degr)
    euc_out  = euclidean(p_clean, p_out)

    # Use shared improvement logic
    cos_stats = compute_improvement(0, cos_degr, cos_out)  # 0 = perfect match
    euc_stats = compute_improvement(0, euc_degr, euc_out)

    return {
        "cosine": {
            "distance_degraded": cos_degr,
            "distance_output": cos_out,
            **cos_stats
        },
        "euclidean": {
            "distance_degraded": euc_degr,
            "distance_output": euc_out,
            **euc_stats
        }
    }


def load_audio(filepath, sr=44100, mono=True):
    y, orig_sr = sf.read(filepath)
    if orig_sr != sr:
        y = librosa.resample(y.T, orig_sr=orig_sr, target_sr=sr).T  # handle stereo resampling
    if mono and y.ndim == 2:
        y = np.mean(y, axis=1)
    return y


# === BAND ENERGY METRIC ===
def band_energy_ratio(y, sr, band_low, band_high, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = np.logical_and(freqs >= band_low, freqs < band_high)
    band_energy = np.sum(S[band, :])
    total_energy = np.sum(S)
    return band_energy / (total_energy + 1e-10)


def compute_improvement(clean, degraded, output):
    err_before = clean - degraded
    err_after = clean - output

    abs_before = abs(err_before)
    abs_after = abs(err_after)

    rel_improvement = (abs_before - abs_after) / (abs_before + 1e-10)
    percent_recovered = 100.0 * (1.0 - abs_after / (abs_before + 1e-10))
    improved = abs_after < abs_before

    rel_error_degraded = abs_before / (abs(clean) + 1e-10)
    rel_error_output   = abs_after / (abs(clean) + 1e-10)

    return {
        "abs_error_before": abs_before,
        "abs_error_after": abs_after,
        "error_before": err_before,
        "error_after": err_after,
        "rel_improvement": rel_improvement,
        "percent_recovered": percent_recovered,
        "rel_error_degraded": rel_error_degraded,
        "rel_error_output": rel_error_output,
        "improved": improved
    }


# === SAMPLE EVALUATION ===
def evaluate_sample(clean, degraded, output, degradations, clean_stereo, degraded_stereo, output_stereo):
    results = {}

    
    for deg in degradations:
        if deg in ["clarity", "bright", "dark", "airy", "warm", "boom", "mud", "vocal"]:
            band_low, band_high = {
                "clarity": (4000, SR // 2),
                "bright":  (6000, SR // 2),
                "dark":    (6000, SR // 2),
                "airy":    (10000, SR // 2),
                "warm":    (20, 400),
                "boom":    (20, 150),
                "mud":    (200, 500),
                "vocal":    (350, 3500),
            }[deg]

            clean_val = band_energy_ratio(clean, SR, band_low, band_high)
            degr_val = band_energy_ratio(degraded, SR, band_low, band_high)
            out_val = band_energy_ratio(output, SR, band_low, band_high)

            stats = compute_improvement(clean_val, degr_val, out_val)

            results[deg] = {
                "ratio_clean": clean_val,
                "ratio_degr": degr_val,
                "ratio_out": out_val,
                **stats
            }


        if deg in ["xband", "mic"]:
            metrics = spectral_balance_metrics(clean, degraded, output, sr=44100)

            results[deg] = {
                "cosine_clean": 0.0,  # perfect alignment = 0
                "cosine_degraded": metrics["cosine"]["distance_degraded"],
                "cosine_output": metrics["cosine"]["distance_output"],
                "cosine_abs_error_before": metrics["cosine"]["abs_error_before"],
                "cosine_abs_error_after": metrics["cosine"]["abs_error_after"],
                "cosine_rel_error_degraded": metrics["cosine"]["rel_error_degraded"],
                "cosine_rel_error_output": metrics["cosine"]["rel_error_output"],
                "cosine_percent_recovered": metrics["cosine"]["percent_recovered"],
                "cosine_rel_improvement": metrics["cosine"]["rel_improvement"],
                "cosine_improved": metrics["cosine"]["improved"],

                "euclidean_clean": 0.0,
                "euclidean_degraded": metrics["euclidean"]["distance_degraded"],
                "euclidean_output": metrics["euclidean"]["distance_output"],
                "euclidean_abs_error_before": metrics["euclidean"]["abs_error_before"],
                "euclidean_abs_error_after": metrics["euclidean"]["abs_error_after"],
                "euclidean_rel_error_degraded": metrics["euclidean"]["rel_error_degraded"],
                "euclidean_rel_error_output": metrics["euclidean"]["rel_error_output"],
                "euclidean_percent_recovered": metrics["euclidean"]["percent_recovered"],
                "euclidean_rel_improvement": metrics["euclidean"]["rel_improvement"],
                "euclidean_improved": metrics["euclidean"]["improved"]
            }


        if deg in ["small", "big", "mix", "real"]:
            rt_clean = estimate_rt60(clean, sr)
            rt_degr = estimate_rt60(degraded, sr)
            rt_out  = estimate_rt60(output, sr)

            if None not in [rt_clean, rt_degr, rt_out]:
                rt_stats = compute_improvement(rt_clean, rt_degr, rt_out)
            else:
                rt_stats = {
                    "abs_error_before": np.nan,
                    "abs_error_after": np.nan,
                    "rel_improvement": np.nan,
                    "percent_recovered": np.nan,
                    "rel_error_degraded": np.nan,
                    "rel_error_output": np.nan,
                    "improved": False
                }

            try:
                msd_degr = modulation_spectrum_distance(clean, degraded, fs=sr)
                msd_out  = modulation_spectrum_distance(clean, output, fs=sr)
                msd_stats = compute_improvement(0.0, msd_degr, msd_out)
            except Exception as e:
                print(f"MSD failed for {deg}: {e}")
                msd_degr = np.nan
                msd_out = np.nan
                msd_stats = {
                    "abs_error_before": np.nan,
                    "abs_error_after": np.nan,
                    "rel_improvement": np.nan,
                    "percent_recovered": np.nan,
                    "rel_error_degraded": np.nan,
                    "rel_error_output": np.nan,
                    "improved": False
                }

            results[deg] = {
                "rt_clean": rt_clean,
                "rt_degraded": rt_degr,
                "rt_output": rt_out,
                **{f"rt60_{k}": v for k, v in rt_stats.items()},  # prefix for clarity

                "modspec_dist_degraded": msd_degr,
                "modspec_dist_output": msd_out,
                **{f"msd_{k}": v for k, v in msd_stats.items()}  # also prefixed
            }


        if deg == "volume":
            rms_clean = rms_energy(clean)
            rms_degr = rms_energy(degraded)
            rms_out  = rms_energy(output)


            stats = compute_improvement(rms_clean, rms_degr, rms_out)


            results[deg] = {
                "rms_clean": rms_clean,
                "rms_degraded": rms_degr,
                "rms_output": rms_out,
                **stats
            }


        if deg == "clip":
            sf_clean = spectral_flatness(clean, sr)
            sf_degr = spectral_flatness(degraded, sr)
            sf_out  = spectral_flatness(output, sr)



            stats = compute_improvement(sf_clean, sf_degr, sf_out)

            results[deg] = {
                "flatness_clean": sf_clean,
                "flatness_degraded": sf_degr,
                "flatness_output": sf_out,
                **stats
            }


        if deg == "stereo":

            r_clean = stereo_energy_ratio(clean_stereo)
            r_degr = stereo_energy_ratio(degraded_stereo)
            r_out  = stereo_energy_ratio(output_stereo)


            stats = compute_improvement(r_clean, r_degr, r_out)


            results[deg] = {
                "stereo_clean": r_clean,
                "stereo_degraded": r_degr,
                "stereo_output": r_out,
                **stats
            }

        if deg == "comp":
            dyn_clean = dynamic_range_std(clean)
            dyn_degr = dynamic_range_std(degraded)
            dyn_out  = dynamic_range_std(output)


            stats = compute_improvement(dyn_clean, dyn_degr, dyn_out)

            results[deg] = {
                "dyn_clean": dyn_clean,
                "dyn_degraded": dyn_degr,
                "dyn_output": dyn_out,
                **stats
            }




        if deg == "punch":
            ts_clean = transient_strength(clean, sr)
            ts_degr = transient_strength(degraded, sr)
            ts_out  = transient_strength(output, sr)




            stats = compute_improvement(ts_clean, ts_degr, ts_out)

            results[deg] = {
                "transient_clean": ts_clean,
                "transient_degraded": ts_degr,
                "transient_output": ts_out,
                **stats
            }

    return results


def summarize_metrics(metrics_by_degradation):
    summary = {}
    summary_multi = {}

    # Set of degradations that have multiple sub-metrics (like reverb types)
    multi_metric_degradations = {"mic", "xband", "small", "big", "mix", "real"}

    for degradation, entries in metrics_by_degradation.items():
        if not entries:
            continue

        if degradation in multi_metric_degradations:
            multi_keys = list(entries[0].keys())
            multi_summary = {}

            for k in multi_keys:
                vals = [x[k] for x in entries]
                if isinstance(vals[0], (int, float, np.number, bool, np.bool_)):
                    avg_val = np.mean(vals)
                    if isinstance(vals[0], (bool, np.bool_)):
                        avg_val = 100.0 * avg_val  # percent
                    multi_summary[f"avg_{k}"] = avg_val

            multi_summary["n_samples"] = len(entries)
            summary_multi[degradation] = multi_summary

        else:
            abs_before = [x["abs_error_before"] for x in entries]
            abs_after = [x["abs_error_after"] for x in entries]
            err_before = [x["error_before"] for x in entries]
            err_after = [x["error_after"] for x in entries]
            rel_improvement = [x["rel_improvement"] for x in entries]
            pct_recovered = [x["percent_recovered"] for x in entries]
            improved = [x["improved"] for x in entries]
            rel_error_degraded = [x["rel_error_degraded"] for x in entries]
            rel_error_output = [x["rel_error_output"] for x in entries]


            summary[degradation] = {
                "n_samples": len(entries),
                "avg_abs_error_before": np.mean(abs_before),
                "avg_abs_error_after": np.mean(abs_after),
                "avg_error_before": np.mean(err_before),
                "std_error_before": np.std(err_before),
                "avg_error_after": np.mean(err_after),
                "std_error_after": np.std(err_after),
                "avg_rel_error_degraded": np.mean(rel_error_degraded),
                "avg_rel_error_output": np.mean(rel_error_output),
                "avg_rel_improvement": np.mean(rel_improvement),
                "avg_percent_recovered": np.mean(pct_recovered),
                "percent_improved": 100.0 * np.mean(improved)
            }

    return summary, summary_multi



# === MAIN PROCESSING LOOP ===
def process_jsonl(jsonl_path, output_dir):
    metrics_by_degradation = defaultdict(list)
    fs=44100
    i=0
    with open(jsonl_path, "r") as f:
        for line in tqdm(f):
            entry = json.loads(line)
            file_id = entry["id"]
            degradations = entry["degradations"]


            # output_dir
            clean_path = entry["original_location"].replace("targetspt","targets").replace(".pt",".flac")
            degraded_path = entry["location"].replace("degradspt","degrads").replace(".pt",".flac")

            filename = os.path.basename(degraded_path)
            output_path = os.path.join(output_dir, filename)



            # if len(degradations)>1:
            #     print("skipped",len(degradations))
            #     continue

            # if len(degradations)<2:
            #     print("skipped",len(degradations))
            #     continue
            
            clean_stereo, degraded_stereo, output_stereo = None, None, None
            # mono=True
            if "stereo" in degradations:
                # mono=False
                clean_stereo = load_audio(clean_path,fs,mono=False)
                degraded_stereo = load_audio(degraded_path,fs,mono=False)
                output_stereo = load_audio(output_path,fs,mono=False)

            try:
                clean = load_audio(clean_path,fs,mono=True)
                degraded = load_audio(degraded_path,fs,mono=True)
                output = load_audio(output_path,fs,mono=True)
            except Exception as e:
                print(f"Failed to load audio for {file_id}: {e}")
                continue

            result = evaluate_sample(clean, degraded, output, degradations, clean_stereo, degraded_stereo, output_stereo)

            for degradation, metrics in result.items():
                metrics_by_degradation[degradation].append(metrics)

            print(i)
            i+=1
            # if i==200:
            #     break

    for k, v in metrics_by_degradation.items():
        print(f"{k}: {len(v)} entries")

    summary, summary_multi = summarize_metrics(metrics_by_degradation)

    return summary, summary_multi


if __name__ == "__main__":
    jsonref = "/testset_pt.jsonl"
    output_csv = "/evaluation/control/attribute_metrics_all.xlsx"

    folders = [
        "outputs/run1",
        "outputs/run2"
    ]

    all_results = []

    for folder in folders:
        print(f"\n🚀 Running evaluation for: {folder}")
        summary, summary_multi = process_jsonl(jsonref, folder)

        for d, m in summary.items():
            m["folder"] = os.path.basename(folder)
            m["degradation"] = d
            all_results.append(m)

        for d, m in summary_multi.items():
            m["folder"] = os.path.basename(folder)
            m["degradation"] = d
            all_results.append(m)

    # ✅ Write to Excel: one sheet per folder, rows = degradations
    with pd.ExcelWriter(output_csv) as writer:
        df_all = pd.DataFrame(all_results)
        for folder, group_df in df_all.groupby("folder"):
            sheet_name = folder[:31]  # Excel limits sheet names to 31 chars
            group_df.drop(columns=["folder"]).to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n✅ All attribute results saved to Excel: {output_csv}")


        # # Save to one big CSV
    # import pandas as pd
    # df = pd.DataFrame(all_results)
    # df.to_csv(output_csv, index=False)
    # print(f"\n✅ Evaluation summary saved to {output_csv}")
