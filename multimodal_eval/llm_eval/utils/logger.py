import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def print_sample_metadata(sample, task, dataset_type):
    logger.info(f"\n🧪 Task: {task.upper()} ({dataset_type})")
    logger.info(f"🖼️ Image: {sample.image}")
    logger.info(f"🤖 Model_Output:  {sample.generated_answer or sample.model_output}")

    if hasattr(sample, "judge_notes") and sample.judge_notes:
        logger.info("\n🔍 [LLM Output]")
        for key, value in sample.judge_notes.items():
            logger.info(f"{key}: {value}")

def print_scores_with_reason(task, scores, model_output=None, sample=None):
    logger.info("\n📊 [LLM Evaluation]")
    for k, v in scores.items():
        logger.info(f"{k}: {v}")
