"""
Run Captioning on Labeled dataset
"""

from multimodal_eval.orchestrator import run_task
from multimodal_eval.configs.tasks_config import CONFIG

if __name__ == "__main__":
    print("🚀 Running Captioning on Labeled dataset...")

    run_task(
        task="captioning",
        dataset_type="labeled",
        task_config=CONFIG["captioning"],
        sources_mode="standard"
    )

    print("✅ Finished. Results saved under results/labeled_data/captioning/")
