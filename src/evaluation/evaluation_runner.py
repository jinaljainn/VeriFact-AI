
import sys
import json
import pandas as pd

from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "evaluation_dataset.csv"
)

REFERENCE_PDF_PATH = (
    PROJECT_ROOT / "data" / "reference_pdfs" / "sample.pdf"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PREDICTIONS_PATH = RESULTS_DIR / "evaluation_results.csv"
SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.json"

LABELS = ["Supported", "Contradicted", "Unverified"]


# --------------------------------------------------
# 2. IMPORT PROJECT MODULES
# --------------------------------------------------

from src.pdf_processing.uploaded_pdf_pipeline import process_uploaded_pdf

from src.hallucination_detection.hallucination_detector import (
    detect_hallucination
)

from src.hallucination_detection.verdict_engine import (
    generate_final_verdict
)


# --------------------------------------------------
# 3. LOAD EVALUATION DATASET
# --------------------------------------------------

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    dataset = pd.read_csv(DATASET_PATH)

    required_columns = {"claim", "expected_verdict"}

    if not required_columns.issubset(dataset.columns):
        raise ValueError(
            "CSV must contain claim and expected_verdict columns."
        )

    dataset = dataset.dropna(
        subset=["claim", "expected_verdict"]
    )

    dataset["claim"] = (
        dataset["claim"].astype(str).str.strip()
    )

    dataset["expected_verdict"] = (
        dataset["expected_verdict"].astype(str).str.strip()
    )

    invalid_labels = (
        set(dataset["expected_verdict"]) - set(LABELS)
    )

    if invalid_labels:
        raise ValueError(
            f"Invalid expected verdicts: {invalid_labels}"
        )

    if dataset.empty:
        raise ValueError("The evaluation dataset is empty.")

    return dataset


# --------------------------------------------------
# 4. PROCESS REFERENCE PDF
# --------------------------------------------------

def prepare_reference_document():

    if not REFERENCE_PDF_PATH.exists():
        raise FileNotFoundError(
            f"Reference PDF not found: {REFERENCE_PDF_PATH}"
        )

    print("\nProcessing reference PDF...")

    with open(REFERENCE_PDF_PATH, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    document_data = process_uploaded_pdf(pdf_bytes)

    print("Reference PDF processed successfully.")

    print(f"Pages: {len(document_data['pages'])}")

    print(
        f"Semantic pieces: "
        f"{len(document_data['semantic_pieces'])}"
    )

    print(
        f"Embeddings: {len(document_data['embeddings'])}"
    )

    return document_data


# --------------------------------------------------
# 5. EVALUATE EACH CLAIM
# --------------------------------------------------

def evaluate_claims(dataset, document_data):

    results = []

    total = len(dataset)

    for index, row in dataset.iterrows():

        claim = row["claim"]
        expected = row["expected_verdict"]

        print(
            f"\nEvaluating claim {len(results) + 1}/{total}"
        )

        print(f"Claim: {claim}")

        verification_results = detect_hallucination(
            claim,
            top_k=5,
            reference_embeddings=document_data["embeddings"]
        )

        final_result = generate_final_verdict(
            claim,
            verification_results
        )

        predicted = final_result["verdict"]

        is_correct = expected == predicted

        results.append({
            "claim": claim,
            "expected_verdict": expected,
            "predicted_verdict": predicted,
            "correct": is_correct
        })

        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"Correct: {is_correct}")

    return pd.DataFrame(results)


# --------------------------------------------------
# 6. CALCULATE EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(results):

    y_true = results["expected_verdict"]
    y_pred = results["predicted_verdict"]

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    )

    summary = {
        "evaluation_date": datetime.now().isoformat(
            timespec="seconds"
        ),

        "total_claims": len(results),

        "correct_predictions": int(
            results["correct"].sum()
        ),

        "incorrect_predictions": int(
            (~results["correct"]).sum()
        ),

        "accuracy": float(accuracy),

        "classification_report": report,

        "confusion_matrix": {
            "labels": LABELS,
            "values": matrix.tolist()
        }
    }

    return summary


# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

def save_results(results, summary):

    results.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\nResults saved successfully:")

    print(PREDICTIONS_PATH)
    print(SUMMARY_PATH)


# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

def display_results(results, summary):

    print("\n" + "=" * 55)

    print("             VERIFACT AI EVALUATION")

    print("=" * 55)

    print("\nIndividual Predictions:")

    print(
        results.to_string(index=False)
    )

    print("\nOverall Metrics:")

    print(
        f"Total Claims: {summary['total_claims']}"
    )

    print(
        f"Correct Predictions: "
        f"{summary['correct_predictions']}"
    )

    print(
        f"Incorrect Predictions: "
        f"{summary['incorrect_predictions']}"
    )

    print(
        f"Accuracy: {summary['accuracy'] * 100:.2f}%"
    )

    print("\nClassification Report:")

    report_df = pd.DataFrame(
        summary["classification_report"]
    ).transpose()

    print(
        report_df.round(3).to_string()
    )

    print("\nConfusion Matrix:")

    print("Label order:", LABELS)

    print(
        pd.DataFrame(
            summary["confusion_matrix"]["values"],
            index=LABELS,
            columns=LABELS
        )
    )

    print("\n" + "=" * 55)


# --------------------------------------------------
# 9. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\nStarting VeriFact AI Evaluation...")

    dataset = load_dataset()

    print(
        f"Loaded {len(dataset)} evaluation claims."
    )

    document_data = prepare_reference_document()

    results = evaluate_claims(
        dataset,
        document_data
    )

    summary = calculate_metrics(results)

    save_results(
        results,
        summary
    )

    display_results(
        results,
        summary
    )


if __name__ == "__main__":
    main()
