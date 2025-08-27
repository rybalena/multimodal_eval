import argparse
from multimodal_eval.orchestrator import run_task
from multimodal_eval.configs.tasks_config import CONFIG

def main():
    parser = argparse.ArgumentParser(description="Run multimodal evaluation")

    parser.add_argument(
        "--task",
        choices=["captioning", "hallucination", "vqa", "contextual_relevance", "all"],
        help="Evaluation task to run"
    )
    parser.add_argument(
        "--dataset_type",
        choices=["labeled", "unlabeled"],
        help="Dataset type: labeled or unlabeled"
    )
    parser.add_argument(
        "--sources",
        choices=["standard", "user", "both"],
        default="standard",
        help="Which datasets to use: only standard, only user_dataset, or both"
    )

    args = parser.parse_args()

    if args.sources == "user" and not args.task and not args.dataset_type:
        # All tasks
        for dtype in ["labeled", "unlabeled"]:
            for task_name, task_config in CONFIG.items():
                print(f"\n🚀 Running USER {dtype.upper()} task: {task_name}")
                run_task(
                    task=task_name,
                    task_config=task_config,
                    dataset_type=dtype,
                    sources_mode="user"
                )
        print("\n✅ All user datasets completed.")
        return

    # Start
    if not args.task or not args.dataset_type:
        parser.error("--task and --dataset_type are required unless running full user mode")

    if args.task == "all":
        for task_name, task_config in CONFIG.items():
            run_task(
                task=task_name,
                task_config=task_config,
                dataset_type=args.dataset_type,
                sources_mode=args.sources
            )
    else:
        run_task(
            task=args.task,
            task_config=CONFIG[args.task],
            dataset_type=args.dataset_type,
            sources_mode=args.sources
        )

if __name__ == "__main__":
    main()
