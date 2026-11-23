
import sys
import json
from pathlib import Path
from datetime import datetime

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = OUTPUT_DIR / "evidence_report.json"

REFERENCE_DOCUMENT = "sample.pdf"


# ============================================================
# IMPORT REQUIRED MODULES
# ============================================================

from src.hallucination_detection.hallucination_detector import (
    detect_hallucination
)

from src.hallucination_detection.verdict_engine import (
    generate_final_verdict
)

from src.hallucination_detection.correction_engine import (
    generate_correction
)


# ============================================================
# HELPER FUNCTION: MAKE DATA JSON SERIALIZABLE
# ============================================================

def make_json_serializable(value):

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    if value is None or isinstance(
        value, (str, int, float, bool)
    ):
        return value

    return str(value)


# ============================================================
# HELPER FUNCTION: FORMAT EVIDENCE
# ============================================================

def format_evidence(evidence_list):

    formatted_evidence = []

    for item in evidence_list:

        formatted_item = {
            "piece_id": item.get(
                "piece_id",
                item.get("id")
            ),

            "page_number": item.get("page_number"),

            "reference": item.get("reference"),

            "similarity_score": item.get(
                "similarity_score"
            ),

            "nli_classification": item.get(
                "nli_classification",
                item.get("classification")
            ),

            "nli_confidence": item.get(
                "nli_confidence",
                item.get("confidence")
            )
        }

        if "scores" in item:
            formatted_item["nli_scores"] = item["scores"]

        elif "classification_scores" in item:
            formatted_item["nli_scores"] = (
                item["classification_scores"]
            )

        formatted_evidence.append(formatted_item)

    return formatted_evidence


# ============================================================
# GENERATE EVIDENCE REPORT
# ============================================================

def generate_evidence_report(claim):

    print("\n" + "=" * 70)
    print("VERIFACT AI - EVIDENCE REPORT GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: SEMANTIC RETRIEVAL AND NLI VERIFICATION
    # --------------------------------------------------------

    print("\nSTEP 1: VERIFYING CLAIM")

    verification_results = detect_hallucination(
        claim=claim,
        top_k=5
    )

    # --------------------------------------------------------
    # STEP 2: GENERATE FINAL VERDICT
    # --------------------------------------------------------

    print("\nSTEP 2: GENERATING FINAL VERDICT")

    final_result = generate_final_verdict(
        claim,
        verification_results
    )

    # --------------------------------------------------------
    # STEP 3: GENERATE CORRECTION
    # --------------------------------------------------------

    print("\nSTEP 3: GENERATING CORRECTION")

    correction_result = generate_correction(
        claim,
        final_result
    )

    # --------------------------------------------------------
    # STEP 4: FORMAT EVIDENCE
    # --------------------------------------------------------

    # Group retrieved evidence using individual NLI labels.
    supporting_raw = [
        item for item in verification_results
        if str(
            item.get(
                "nli_classification",
                item.get("classification", "")
            )
        ).strip().lower() == "entailment"
    ]

    contradicting_raw = [
        item for item in verification_results
        if str(
            item.get(
                "nli_classification",
                item.get("classification", "")
            )
        ).strip().lower() == "contradiction"
    ]

    neutral_raw = [
        item for item in verification_results
        if str(
            item.get(
                "nli_classification",
                item.get("classification", "")
            )
        ).strip().lower() == "neutral"
    ]

    supporting_evidence = format_evidence(supporting_raw)

    contradicting_evidence = format_evidence(
        contradicting_raw
    )

    neutral_evidence = format_evidence(neutral_raw)

    # --------------------------------------------------------
    # STEP 5: PREPARE COMPLETE REPORT
    # --------------------------------------------------------

    report = {

        "project": "VeriFact AI",

        "report_generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),

        "reference_document": REFERENCE_DOCUMENT,

        "original_claim": claim,

        "final_verdict": {
            "status": final_result["verdict"],
            "explanation": final_result.get("explanation")
        },

        "correction": {

            "corrected_claim": correction_result.get(
                "corrected_claim"
            ),

            "correction_source": correction_result.get(
                "correction_source"
            ),

            "source_page": correction_result.get(
                "source_page"
            ),

            "source_passage": correction_result.get(
                "source_passage"
            ),

            "explanation": correction_result.get(
                "explanation"
            )
        },

        "evidence_summary": {

            "total_references_checked": len(
                verification_results
            ),

            "supporting_evidence_count": len(
                supporting_evidence
            ),

            "contradicting_evidence_count": len(
                contradicting_evidence
            ),

            "neutral_evidence_count": len(
                neutral_evidence
            ),

            "supporting_evidence": supporting_evidence,

            "contradicting_evidence": contradicting_evidence,

            "neutral_evidence": neutral_evidence
        },

        "all_verification_results": verification_results
    }

    report = make_json_serializable(report)

    # --------------------------------------------------------
    # STEP 6: SAVE REPORT AS JSON
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # STEP 7: DISPLAY REPORT SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVIDENCE REPORT SUMMARY")
    print("=" * 70)

    print("\nOriginal claim:")
    print(claim)

    print("\nFinal verdict:")
    print(final_result["verdict"])

    print("\nCorrected claim:")
    print(
        correction_result.get("corrected_claim")
        or "No correction available."
    )

    print("\nEvidence statistics:")

    print(
        "Total references checked:",
        len(verification_results)
    )

    print(
        "Supporting evidence:",
        len(supporting_evidence)
    )

    print(
        "Contradicting evidence:",
        len(contradicting_evidence)
    )

    print(
        "Neutral evidence:",
        len(neutral_evidence)
    )

    print("\nReport saved successfully!")

    print("File location:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("VERIFACT AI - REPORT GENERATION COMPLETED!")
    print("=" * 70)

    return report


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    sample_claim = (
        "The capital of Australia is Canberra."
    )

    generate_evidence_report(sample_claim)
