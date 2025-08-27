import torch
import spacy
from sentence_transformers import SentenceTransformer, util
from multimodal_eval.evaluation.schema import Sample
from multimodal_eval.utils.constants import NEUTRAL_WORDS

#  POS tag weights (importance per part of speech)
POS_WEIGHTS = {
    "NOUN": 1.0,
    "ADJ": 0.3,
    "VERB": 0.8
}

# Similarity thresholds for semantic comparisons
SIMILARITY_THRESHOLD = 0.75
NEUTRAL_SIMILARITY_THRESHOLD = 0.8

#  Smoothing to reduce sensitivity
HALLUCINATION_SMOOTHING = 0.1

#  Final score weights
HALLUC_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6

# Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", device=str(device))


def extract_tokens(text):
    """
    Tokenizes the text and filters tokens by POS and neutral words list.
    Returns a list of (word, POS) tuples.
    """
    doc = nlp(text)
    return [
        (token.text.lower(), token.pos_)
        for token in doc
        if token.pos_ in POS_WEIGHTS and token.text.lower() not in NEUTRAL_WORDS
    ]


def is_similar_to_neutral(word: str) -> bool:
    """Checks whether the word is semantically similar to a neutral word."""
    sims = util.cos_sim(
        sbert_model.encode(word, convert_to_tensor=True),
        sbert_model.encode(list(NEUTRAL_WORDS), convert_to_tensor=True)
    )[0]
    return bool(torch.any(sims >= NEUTRAL_SIMILARITY_THRESHOLD))


def semantic_score(from_tokens, to_tokens):
    """
    Calculates hallucination score between two sets of tokens.
    Returns a score and a list of hallucinated words.
    """
    total_weight = 0
    mismatch_weight = 0
    hallucinated_words = []

    grouped_to = {pos: [word for word, p in to_tokens if p == pos] for pos in POS_WEIGHTS}

    for word, pos in from_tokens:
        weight = POS_WEIGHTS.get(pos, 0.5)
        total_weight += weight
        candidates = grouped_to.get(pos, [])

        if not candidates:
            if not is_similar_to_neutral(word):
                mismatch_weight += weight
                hallucinated_words.append(word)
            continue

        sims = util.cos_sim(
            sbert_model.encode(word, convert_to_tensor=True),
            sbert_model.encode(candidates, convert_to_tensor=True)
        )[0]

        if max(sims) >= SIMILARITY_THRESHOLD:
            continue

        if is_similar_to_neutral(word):
            continue

        mismatch_weight += weight
        hallucinated_words.append(word)

    score = mismatch_weight / total_weight if total_weight > 0 else 0.0
    return score, hallucinated_words


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Computes cosine similarity between two texts using SBERT.
    """
    emb1 = sbert_model.encode(text1, convert_to_tensor=True)
    emb2 = sbert_model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    return round(similarity, 4)


def compute_hallucination_metric(sample: Sample) -> dict:
    """
    Main hallucination evaluation function.

    Compares model_output to caption to detect hallucinated content.
    Returns:
        - hallucinated_objects: list of hallucinated words
        - hallucination_raw: raw score before smoothing (higher = worse)
        - hallucination_score: 1 - hallucination_raw (higher = better)
        - semantic_similarity: cosine similarity between full texts
        - composite_score: weighted combination of hallucination and semantic scores

    """
    caption = (sample.caption or "").strip()
    model_output = (sample.model_output or "").strip()
    sample_id = getattr(sample, "id", "unknown")

    if not caption or not model_output:
        print(f"[⚠️ Empty input] caption or model_output is missing in sample {sample_id}")
        return {
            "hallucinated_objects": [],
            "hallucination_raw": 0.0,
            "hallucination_score": 1.0,
            "semantic_similarity": 0.0,
            "composite_score": 0.5
        }

    try:
        caption_tokens = extract_tokens(caption)
        output_tokens = extract_tokens(model_output)

        # Raw score
        hallucination_raw, hallucinated = semantic_score(output_tokens, caption_tokens)

        hallucination_score = max(0.0, min(1.0, 1 - hallucination_raw))

        # Semantic similarity
        semantic_similarity = compute_semantic_similarity(caption, model_output)

        composite_score = round(
            (1 - max(0.0, hallucination_raw - HALLUCINATION_SMOOTHING)) * HALLUC_WEIGHT
            + semantic_similarity * SEMANTIC_WEIGHT,
            4
        )

        # Debug output
        print(f"❗ Hallucinated: {hallucinated}")
        print(f"📏 Hallucination Raw: {round(hallucination_raw, 4)}")
        print(f"✅ Hallucination Score: {round(hallucination_score, 4)}")
        print(f"🧠 Semantic Similarity: {semantic_similarity}")
        print(f"📊 Composite Score: {composite_score}")

        return {
            "hallucinated_objects": hallucinated,
            "hallucination_raw": round(hallucination_raw, 4),
            "hallucination_score": round(hallucination_score, 4),
            "semantic_similarity": semantic_similarity,
            "composite_score": composite_score
        }

    except Exception as e:
        import traceback
        print(f"[Hallucination Error] {sample_id}: {e}")
        traceback.print_exc()
        return {
            "hallucinated_objects": [],
            "hallucination_raw": 1.0,
            "hallucination_score": 0.0,
            "semantic_similarity": 0.0,
            "composite_score": 0.0
        }
