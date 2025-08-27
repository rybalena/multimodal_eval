from multimodal_eval.configs.tasks_config import CONFIG
from multimodal_eval.orchestrator import run_task

if __name__ == "__main__":
    # Run tasks on the LABELED dataset
    for task_name in CONFIG:
        print(f"\n🚀 Running LABELED task: {task_name}")
        run_task(task_name, CONFIG[task_name], dataset_type="labeled", sources_mode="standard")

    # Run tasks on the UNLABELED dataset
    for task_name in CONFIG:
        print(f"\n🚀 Running UNLABELED task: {task_name}")
        run_task(task_name, CONFIG[task_name], dataset_type="unlabeled", sources_mode="standard")
