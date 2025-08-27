"""
Evaluator — the core evaluation logic.
Applies selected metrics to a list of Samples.
"""

from typing import List, Dict, Any, Optional, Union
from multimodal_eval.evaluation.schema import Sample
from multimodal_eval.evaluation import metrics_registry
from multimodal_eval.model_wrappers.model_registry import get_model_wrapper


class Evaluator:
    def __init__(self, metrics: Optional[List[str]] = None, model_name: str = "llava"):
        """
        :param metrics: list of metrics (e.g., ["clip_score", "hallucination"]).
                        If None, all registered metrics will be used.
        :param model_name: model name to use for generation (e.g., "llava", "qwen", "openai").
        """
        self.metrics = metrics or metrics_registry.list_available()
        self.model = get_model_wrapper(model_name)

    def maybe_generate_output(self, sample: Sample) -> Sample:
        """
        Generates output for a sample if it's missing `generated_answer` and has an image.
        Uses either the question or a default prompt.
        """
        if not sample.generated_answer and sample.image:
            try:
                if sample.question:
                    answer = self.model.generate(prompt=sample.question, image_path=sample.image)
                    sample.generated_answer = answer
                else:
                    caption = self.model.generate(prompt="Describe this image", image_path=sample.image)
                    sample.generated_answer = caption
            except Exception as e:
                print(f"[Evaluator] Failed to generate output for sample {sample.id}: {e}")
        return sample

    def evaluate(
        self,
        samples: List[Sample],
        per_sample: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates a list of samples using selected metrics.

        :param samples: list of Sample objects
        :param per_sample: if True, returns detailed scores per sample
        :return: dictionary with summary scores and optional per-sample breakdown
        """
        if not samples:
            raise ValueError("Empty list of samples provided to Evaluator")

        results = {}
        per_sample_results = {metric: [] for metric in self.metrics} if per_sample else None

        for metric_name in self.metrics:
            metric_fn = metrics_registry.get_metric(metric_name)
            all_scores = []

            for sample in samples:
                sample = self.maybe_generate_output(sample)

                if not sample.generated_answer:
                    print(f"[Evaluator] Sample {sample.id} is missing generated_answer. Skipping.")
                    continue

                try:
                    score: Union[float, dict] = metric_fn(sample)
                except Exception as e:
                    print(f"[Evaluator] Error on sample {sample.id} with metric {metric_name}: {e}")
                    score = 0.0

                all_scores.append(score)

                if per_sample:
                    if isinstance(score, dict):
                        per_sample_results[metric_name].append({
                            "id": sample.id,
                            **score
                        })
                    else:
                        per_sample_results[metric_name].append({
                            "id": sample.id,
                            "score": round(score, 4)
                        })

            if all(isinstance(s, (float, int)) for s in all_scores):
                avg_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
            else:
                avg_score = None

            results[metric_name] = avg_score

        return {
            "summary": results,
            "per_sample": per_sample_results if per_sample else None
        }
