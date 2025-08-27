"""
Show metrics for Captioning task (Labeled and Unlabeled)
"""

import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    results_file = project_root / "results" / "all_runs.csv"

    if not results_file.exists():
        print(f"❌ No results found at {results_file}")
    else:
        df = pd.read_csv(results_file)

        # normalize text
        df["task"] = df["task"].str.strip().str.lower()
        df["dataset_type"] = df["dataset_type"].str.strip().str.lower()

        captioning_df = df[df["task"] == "captioning"]

        if captioning_df.empty:
            print("⚠️ No captioning runs found.")
        else:
            # --- Labeled ---
            labeled = captioning_df[captioning_df["dataset_type"] == "labeled_data"]
            if not labeled.empty:
                print("\n📊 Captioning (Labeled) Results:")
                print(labeled[
                    ["run_timestamp","model_name","mean_cider","mean_clip_score","mean_semantic_similarity","mean_composite_score"]
                ].tail(10).to_string(index=False))

            # --- Unlabeled ---
            unlabeled = captioning_df[captioning_df["dataset_type"] == "unlabeled_data"]
            if not unlabeled.empty:
                print("\n📊 Captioning (Unlabeled) Results:")
                print(unlabeled[
                    ["run_timestamp","model_name","mean_image_grounding","mean_coverage","mean_fluency","mean_composite_score"]
                ].tail(10).to_string(index=False))
