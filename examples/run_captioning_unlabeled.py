"""
Run Captioning on Unlabeled dataset
"""

from multimodal_eval.orchestrator import run_task
from multimodal_eval.configs.tasks_config import CONFIG

if __name__ == "__main__":
    print("🚀 Running Captioning on Unlabeled dataset...")

    run_task(
        task="captioning",
        dataset_type="unlabeled",
        task_config=CONFIG["captioning"],
        sources_mode="standard"
    )

    print("✅ Finished. Results saved under results/unlabeled_data/captioning/")
