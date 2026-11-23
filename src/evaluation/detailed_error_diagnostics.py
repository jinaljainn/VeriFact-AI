
import sys
import json
import pandas as pd

from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

EVALUATION_RESULTS_PATH = (
    PROJECT_ROOT / "results" / "evaluation_results.csv"
)

REFERENCE_PDF_PATH = (
    PROJECT_ROOT / "data" / "reference_pdfs" / "sample.pdf"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "detailed_error_diagnostics.json"
)


# --------------------------------------------------
# 2. IMPORT PROJECT MODULES
# --------------------------------------------------

from src.pdf_processing.uploaded_pdf_pipeline import (
    process_uploaded_pdf
)

from src.hallucination_detection.hallucination_detector import (
    detect_hallucination
)

from src.hallucination_detection.verdict_engine import (
    generate_final_verdict
)


# --------------------------------------------------
# 3. LOAD INCORRECT PREDICTIONS
# --------------------------------------------------

def load_incorrect_predictions():

    if not EVALUATION_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Evaluation results not found. "
            "Run evaluation_runner.py first."
        )

    results = pd.read_csv(EVALUATION_RESULTS_PATH)

    errors = results[
        results["correct"].astype(str).str.lower() == "false"
    ]

    if errors.empty:
        print("No incorrect predictions found.")
        return errors

    return errors


# --------------------------------------------------
# 4. PROCESS REFERENCE DOCUMENT
# --------------------------------------------------

def load_reference_document():

    if not REFERENCE_PDF_PATH.exists():
        raise FileNotFoundError(
            f"Reference PDF not found: {REFERENCE_PDF_PATH}"
        )

    print("\nProcessing reference PDF...")

    with open(REFERENCE_PDF_PATH, "rb") as file:
        pdf_bytes = file.read()

    document_data = process_uploaded_pdf(pdf_bytes)

    print("Reference PDF loaded successfully.")

    return document_data


# --------------------------------------------------
# 5. RUN DETAILED DIAGNOSTICS
# --------------------------------------------------

def analyse_claims(errors, document_data):

    diagnostic_records = []

    for index, (_, row) in enumerate(
        errors.iterrows(),
        start=1
    ):

        claim = row["claim"]

        print("\n" + "=" * 70)
        print(f"DIAGNOSTIC {index}/{len(errors)}")
        print("=" * 70)

        print(f"\nClaim: {claim}")

        print(
            f"Expected verdict: {row['expected_verdict']}"
        )

        print(
            f"Previous predicted verdict: "
            f"{row['predicted_verdict']}"
        )

        # Run semantic retrieval and NLI again
        verification_results = detect_hallucination(
            claim,
            top_k=5,
            reference_embeddings=document_data["embeddings"]
        )

        # Generate final verdict
        final_result = generate_final_verdict(
            claim,
            verification_results
        )

        # Store retrieved evidence
        retrieved_evidence = []

        for item in verification_results:

            evidence = {
                "rank": item.get("rank"),
                "piece_id": item.get("piece_id"),
                "page_number": item.get("page_number"),
                "reference": item.get("reference"),
                "similarity_score": item.get(
                    "similarity_score"
                ),
                "classification": item.get(
                    "classification"
                ),
                "confidence": item.get("confidence"),
                "scores": item.get("scores")
            }

            retrieved_evidence.append(evidence)

            print("\nRetrieved reference:")
            print(
                f"Page: {evidence['page_number']}"
            )
            print(
                f"Similarity: {evidence['similarity_score']}"
            )
            print(
                f"Classification: {evidence['classification']}"
            )
            print(
                f"Confidence: {evidence['confidence']}"
            )
            print(
                f"Text: {evidence['reference']}"
            )

        # Store final verdict information
        record = {
            "claim": claim,
            "expected_verdict": row["expected_verdict"],
            "previous_predicted_verdict": row[
                "predicted_verdict"
            ],
            "diagnostic_predicted_verdict": final_result[
                "verdict"
            ],
            "explanation": final_result.get("explanation"),
            "retrieved_evidence": retrieved_evidence,
            "supporting_evidence": final_result.get(
                "supporting_evidence", []
            ),
            "contradicting_evidence": final_result.get(
                "contradicting_evidence", []
            ),
            "neutral_evidence": final_result.get(
                "neutral_evidence", []
            ),
            "exact_match_evidence": final_result.get(
                "exact_match_evidence", []
            )
        }

        diagnostic_records.append(record)

        print("\nFinal diagnostic verdict:")
        print(record["diagnostic_predicted_verdict"])

    return diagnostic_records


# --------------------------------------------------
# 6. SAVE DIAGNOSTIC REPORT
# --------------------------------------------------

def save_diagnostics(records):

    report = {
        "project": "VeriFact AI",
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "total_errors_analysed": len(records),
        "diagnostics": records
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nDetailed diagnostic report saved:")
    print(OUTPUT_PATH)


# --------------------------------------------------
# 7. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\nStarting Detailed Error Diagnostics...")

    errors = load_incorrect_predictions()

    if errors.empty:
        return

    document_data = load_reference_document()

    records = analyse_claims(
        errors,
        document_data
    )

    save_diagnostics(records)

    print("\nDetailed error diagnostics completed!")


if __name__ == "__main__":
    main()
