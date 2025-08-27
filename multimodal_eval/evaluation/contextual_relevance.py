import torch
from sentence_transformers import SentenceTransformer
from multimodal_eval.evaluation.schema import Sample

# Load SBERT model once
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
_MODEL.eval()

def compute_contextual_relevance(sample: Sample) -> float:
    """
    Compares the generated_answer with the context:
    - If prompt_template / prompt / question / expected_answer exist → builds context from them
    - If only expected_answer exists → compares expected vs. generated (semantic similarity fallback)

    Returns a score in the range [0.0, 1.0]
    """
    # Extract all fields
    prompt = sample.prompt_template or sample.prompt or ""
    question = sample.question or ""
    expected = sample.expected_answer or ""
    generated = sample.generated_answer or ""

    # Build context
    context_parts = [prompt.strip(), question.strip(), expected.strip()]
    context = " ".join(part for part in context_parts if part)

    # Fallback to expected if context is empty
    if not context and expected:
        context = expected

    if not context or not generated.strip():
        return 0.0

    try:
        embeddings = _MODEL.encode([context, generated], convert_to_tensor=True)
        cosine_sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
        return round((cosine_sim + 1) / 2, 4)
    except Exception as e:
        print(f"[Contextual Relevance Error] sample_id={sample.id} → {e}")
        return 0.0
