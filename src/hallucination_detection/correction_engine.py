import sys
from pathlib import Path

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT REQUIRED MODULES
# ============================================================

from src.hallucination_detection.hallucination_detector import (
    detect_hallucination
)

from src.hallucination_detection.verdict_engine import (
    generate_final_verdict
)


# ============================================================
# EVIDENCE SELECTION
# ============================================================

def select_best_correcting_evidence(contradicting_evidence):
    """
    Select the most relevant contradicting passage.

    Priority:
    1. Higher similarity to the original claim.
    2. Higher NLI confidence.
    3. Original order when both scores are tied.

    The selected passage is copied exactly from the reference
    document. This function does not invent or rewrite facts.
    """

    usable_evidence = []

    for item in contradicting_evidence:
        reference = item.get("reference")

        if not isinstance(reference, str) or not reference.strip():
            continue

        try:
            similarity = float(item.get("similarity_score", 0.0))
            confidence = float(item.get("nli_confidence", 0.0))
        except (TypeError, ValueError):
            continue

        usable_evidence.append((item, similarity, confidence))

    if not usable_evidence:
        return None

    best_item, _, _ = max(
        usable_evidence,
        key=lambda entry: (entry[1], entry[2])
    )

    return best_item


# ============================================================
# CORRECTION ENGINE
# ============================================================

def generate_correction(claim, final_result):

    verdict = final_result["verdict"]

    correction = {
        "original_claim": claim,
        "verdict": verdict,
        "corrected_claim": None,
        "correction_source": None,
        "source_page": None,
        "source_passage": None,
        "explanation": None
    }

    # --------------------------------------------------------
    # CASE 1: CLAIM IS CONTRADICTED
    # --------------------------------------------------------

    if verdict == "Contradicted":

        contradicting_evidence = (
            final_result.get("contradicting_evidence", [])
        )

        best_evidence = select_best_correcting_evidence(
            contradicting_evidence
        )

        if best_evidence is None:

            correction["explanation"] = (
                "The claim is contradicted, but no usable "
                "correcting evidence is available."
            )

            return correction

        # Preserve the exact wording from the reference document.
        correction["corrected_claim"] = (
            best_evidence["reference"]
        )

        correction["source_passage"] = (
            best_evidence["reference"]
        )

        correction["correction_source"] = (
            "Reference document"
        )

        correction["source_page"] = (
            best_evidence.get("page_number")
        )

        correction["explanation"] = (
            "The original claim conflicts with the reference "
            "document. The corrected statement is copied exactly "
            "from the most relevant contradicting passage. "
            "No new factual information has been added."
        )

    # --------------------------------------------------------
    # CASE 2: CLAIM IS SUPPORTED
    # --------------------------------------------------------

    elif verdict == "Supported":

        correction["corrected_claim"] = claim

        correction["correction_source"] = (
            "Original claim"
        )

        correction["explanation"] = (
            "The claim is supported by the available reference "
            "evidence. No correction is necessary."
        )

    # --------------------------------------------------------
    # CASE 3: CLAIM IS UNVERIFIED
    # --------------------------------------------------------

    else:

        correction["explanation"] = (
            "The reference document does not provide enough "
            "reliable evidence to correct this claim. "
            "A correction cannot be generated safely."
        )

    return correction


# ============================================================
# DISPLAY CORRECTION RESULT
# ============================================================

def display_correction(correction):

    print("\n" + "=" * 70)
    print("VERIFACT AI - HALLUCINATION CORRECTION")
    print("=" * 70)

    print("\nORIGINAL CLAIM:")
    print(correction["original_claim"])

    print("\nFINAL VERDICT:")
    print(correction["verdict"])

    print("\nCORRECTION EXPLANATION:")
    print(correction["explanation"])

    print("\n" + "-" * 70)
    print("CORRECTED STATEMENT")
    print("-" * 70)

    if correction["corrected_claim"]:

        print(correction["corrected_claim"])

        print(
            "\nCorrection source:",
            correction["correction_source"]
        )

        if correction["source_page"] is not None:

            print(
                "Reference page:",
                correction["source_page"]
            )

        if correction["source_passage"]:

            print("\nEXACT SOURCE PASSAGE:")
            print(correction["source_passage"])

    else:

        print(
            "No correction suggested because "
            "sufficient evidence is unavailable."
        )

    print("\n" + "=" * 70)
    print("CORRECTION PROCESS COMPLETED!")
    print("=" * 70)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    sample_claim = (
        "Correlation reveals how strongly a pair of variables "
    "are related to each other and change together."
    )

    print("\nRunning hallucination detection...")

    # STEP 1: Retrieve evidence and perform NLI verification
    verification_results = detect_hallucination(
        claim=sample_claim,
        top_k=5
    )

    # STEP 2: Generate final verdict
    final_result = generate_final_verdict(
        sample_claim,
        verification_results
    )

    # STEP 3: Generate evidence-grounded correction
    correction_result = generate_correction(
        sample_claim,
        final_result
    )

    # STEP 4: Display correction
    display_correction(correction_result)
