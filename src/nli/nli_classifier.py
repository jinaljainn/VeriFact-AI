
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

MODEL_REPO = "Xenova/nli-deberta-v3-xsmall"

CACHE_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Xenova--nli-deberta-v3-xsmall"
    / "snapshots"
)

LABELS = {
    0: "Contradiction",
    1: "Entailment",
    2: "Neutral"
}


# --------------------------------------------------
# FIND DOWNLOADED MODEL
# --------------------------------------------------

def find_model_directory():

    if not CACHE_DIR.exists():
        raise FileNotFoundError(
            "NLI model cache directory was not found."
        )

    for snapshot in CACHE_DIR.iterdir():

        model_file = (
            snapshot / "onnx" / "model_quantized.onnx"
        )

        tokenizer_file = snapshot / "tokenizer.json"
        config_file = snapshot / "config.json"

        if (
            model_file.exists()
            and tokenizer_file.exists()
            and config_file.exists()
        ):
            return snapshot

    raise FileNotFoundError(
        "Required ONNX model or tokenizer files were not found."
    )


# --------------------------------------------------
# LOAD MODEL AND TOKENIZER
# --------------------------------------------------

print("Loading NLI model...")

MODEL_DIR = find_model_directory()

tokenizer = Tokenizer.from_file(
    str(MODEL_DIR / "tokenizer.json")
)

session = ort.InferenceSession(
    str(MODEL_DIR / "onnx" / "model_quantized.onnx"),
    providers=["CPUExecutionProvider"]
)

print("NLI model loaded successfully!")


# --------------------------------------------------
# NLI CLASSIFICATION
# --------------------------------------------------

def classify_claim(premise, hypothesis):

    # Tokenize reference evidence and claim together
    encoded = tokenizer.encode(
        premise,
        hypothesis
    )

    input_ids = np.array(
        [encoded.ids],
        dtype=np.int64
    )

    attention_mask = np.array(
        [encoded.attention_mask],
        dtype=np.int64
    )

    # Run ONNX inference
    outputs = session.run(
        ["logits"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    )

    logits = outputs[0][0]

    # Convert logits into probabilities
    exp_scores = np.exp(
        logits - np.max(logits)
    )

    probabilities = exp_scores / np.sum(exp_scores)

    predicted_index = int(
        np.argmax(probabilities)
    )

    return {
        "classification": LABELS[predicted_index],
        "confidence": round(
            float(probabilities[predicted_index]) * 100,
            2
        ),
        "scores": {
            LABELS[index]: round(
                float(probabilities[index]) * 100,
                2
            )
            for index in range(3)
        }
    }


# --------------------------------------------------
# TEST THE NLI CLASSIFIER
# --------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("VERIFACT AI - NLI CLASSIFICATION")
    print("=" * 60)

    reference = (
        "Correlation identifies patterns of association "
        "but does not imply one variable affects the other."
    )

    claim = (
        "Correlation always means that one variable "
        "causes a change in another variable."
    )

    print("\nREFERENCE EVIDENCE:")
    print(reference)

    print("\nCLAIM TO VERIFY:")
    print(claim)

    result = classify_claim(
        reference,
        claim
    )

    print("\n" + "=" * 60)
    print("NLI CLASSIFICATION RESULT")
    print("=" * 60)

    print("\nPredicted classification:")
    print(result["classification"])

    print("\nConfidence:")
    print(f"{result['confidence']}%")

    print("\nAll classification scores:")

    for label, score in result["scores"].items():
        print(f"{label}: {score}%")

    print("\n" + "=" * 60)
    print("NLI CLASSIFICATION COMPLETED!")
    print("=" * 60)
