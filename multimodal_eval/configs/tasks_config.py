CONFIG = {
    "captioning": {
        "labeled": "data_sets/labeled_data/captioning/captioning_set.json",
        "unlabeled": "data_sets/unlabeled_data/captioning/captioning_set.json",

        "model_output_labeled": "model_outputs/labeled_data/captioning/labeled_captioning_predictions.json",
        "model_output_unlabeled": "model_outputs/unlabeled_data/captioning/captioning_predictions.json",

        "metrics": [
            "cider",
            "clip_score",
            "semantic_similarity"
        ]
    },

    "hallucination": {
        "labeled": "data_sets/labeled_data/hallucination/hallucination_set.json",
        "unlabeled": "data_sets/unlabeled_data/hallucination/hallucination_set.json",

        "model_output_labeled": "model_outputs/labeled_data/hallucination/labeled_hallucination_predictions.json",
        "model_output_unlabeled": "model_outputs/unlabeled_data/hallucination/hallucination_predictions.json",

        "metrics": [
            "hallucination_metric"
        ]
    },

    "vqa": {
        "labeled": "data_sets/labeled_data/vqa/vqa_set.json",

        "model_output_labeled": "model_outputs/labeled_data/vqa/labeled_vqa_predictions.json",

        "metrics": [
            "clip_score",
            "semantic_similarity",
            "contextual_relevance"
        ]
    },

    "contextual_relevance": {
        "labeled": "data_sets/labeled_data/contextual_relevance/contextual_relevance_set.json",
        "unlabeled": "data_sets/unlabeled_data/contextual_relevance/contextual_relevance_set.json",

        "model_output_labeled": "model_outputs/labeled_data/contextual_relevance/contextual_relevance_predictions.json",
        "model_output_unlabeled": "model_outputs/unlabeled_data/contextual_relevance/contextual_relevance_predictions.json",


        "metrics": [
            "clip_score",
            "cider",
            "semantic_similarity"
        ]
    }
}
