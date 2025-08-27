import torch
from sentence_transformers import SentenceTransformer
from multimodal_eval.evaluation.schema import Sample

_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
_MODEL.eval()

def compute_semantic_similarity(sample: Sample) -> float:
    """
    Computes semantic similarity between expected_answer or caption and generated_answer.

    :param sample: Sample with fields expected_answer / caption and generated_answer
    :return: float in range [0.0, 1.0]
    """
    reference = sample.expected_answer or sample.caption
    hypothesis = sample.generated_answer

    if not reference or not hypothesis:
        return 0.0

    try:
        embeddings = _MODEL.encode([reference.strip(), hypothesis.strip()], convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
        return round((similarity + 1) / 2, 4)
    except Exception as e:
        print(f"[SemanticSim Error] sample_id={sample.id}: {e}")
        return 0.0
