import os
import json
import pandas as pd
from audiobox_aesthetics.infer import initialize_predictor
from tqdm import tqdm

predictor = initialize_predictor()

def categorize_and_score(folder, jsonl_path):
    group_scores = {
        "single": [],
        "multiple": [],
        "all": []
    }

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc=os.path.basename(folder)):
            entry = json.loads(line)
            sample_id = entry["id"]
            degradations = entry.get("degradations", [])

            if not isinstance(degradations, list):
                print(f"⚠️ Skipping malformed entry: {sample_id}")
                continue

            group = "multiple" if len(degradations) > 1 else "single"

            file_path = os.path.join(folder, f"{sample_id}"+".flac")

            try:
                result = predictor.forward([{"path": file_path}])[0]
                group_scores[group].append(result)
                group_scores["all"].append(result)
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
                continue

    return {
        k: pd.DataFrame(v).mean().to_dict() if v else {} 
        for k, v in group_scores.items()
    }

if __name__ == "__main__":
    jsonl_path = "/testset_pt.jsonl"
    output_csv = "/evaluationfinal/aesthetic_summary_models.csv"

    folders = [
        "outputs/run1",
        "outputs/run2"
    ]

    summary = []

    for folder in folders:
        print(f"\n🚀 Processing: {folder}")
        grouped_results = categorize_and_score(folder, jsonl_path)
        for group_name, metrics in grouped_results.items():
            if metrics:  # skip if no results
                row = {"folder": os.path.basename(folder), "group": group_name}
                row.update(metrics)
                summary.append(row)

    df = pd.DataFrame(summary)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved grouped aesthetic scores to: {output_csv}")
    print(df)
