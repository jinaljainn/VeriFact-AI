
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import snapshot_download


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

MODEL_REPO = "Xenova/nli-deberta-v3-xsmall"

LABELS = {
    0: "Contradiction",
    1: "Entailment",
    2: "Neutral"
}


# --------------------------------------------------
# DOWNLOAD MODEL AUTOMATICALLY
# --------------------------------------------------

def download_model():

    print("Loading NLI model...")

    model_dir = snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=[
            "onnx/model_quantized.onnx",
            "tokenizer.json"
        ]
    )

    return Path(model_dir)


# --------------------------------------------------
# LOAD MODEL AND TOKENIZER
# --------------------------------------------------

MODEL_DIR = download_model()

model_file = MODEL_DIR / "onnx" / "model_quantized.onnx"
tokenizer_file = MODEL_DIR / "tokenizer.json"

if not model_file.exists():
    raise FileNotFoundError(
        "NLI ONNX model file was not downloaded."
    )

if not tokenizer_file.exists():
    raise FileNotFoundError(
        "NLI tokenizer file was not downloaded."
    )

tokenizer = Tokenizer.from_file(
    str(tokenizer_file)
)

session = ort.InferenceSession(
    str(model_file),
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
