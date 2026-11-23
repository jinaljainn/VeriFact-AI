
import sys
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# IMPORT MODULES
# --------------------------------------------------

from src.embeddings.semantic_retrieval import (
    retrieve_similar_pieces,
    retrieve_similar_pieces_from_embeddings
)

from src.nli.nli_classifier import classify_claim


# --------------------------------------------------
# HALLUCINATION DETECTION
# --------------------------------------------------

def detect_hallucination(
    claim,
    top_k=5,
    reference_embeddings=None
):
    """
    Detect potential hallucinations using semantic retrieval
    and Natural Language Inference.

    If reference_embeddings are provided, the function
    searches the uploaded document.

    Otherwise, it uses the original saved embeddings.json.
    """

    print("\n" + "=" * 65)
    print("VERIFACT AI - HALLUCINATION DETECTION")
    print("=" * 65)

    print("\nCLAIM TO VERIFY:")
    print(claim)

    # --------------------------------------------------
    # STEP 1: SEMANTIC RETRIEVAL
    # --------------------------------------------------

    print("\nSTEP 1: SEMANTIC RETRIEVAL")
    print("-" * 65)

    if reference_embeddings is not None:

        print("\nUsing uploaded document embeddings...")

        retrieved_pieces = retrieve_similar_pieces_from_embeddings(
            query=claim,
            reference_embeddings=reference_embeddings,
            top_k=top_k
        )

    else:

        print("\nUsing saved reference embeddings...")

        retrieved_pieces = retrieve_similar_pieces(
            claim,
            top_k=top_k
        )

    print(
        f"\nTotal reference pieces retrieved: "
        f"{len(retrieved_pieces)}"
    )

    if not retrieved_pieces:
        raise ValueError(
            "No relevant reference pieces were retrieved."
        )

    # --------------------------------------------------
    # STEP 2: NLI VERIFICATION
    # --------------------------------------------------

    print("\nSTEP 2: NLI VERIFICATION")
    print("-" * 65)

    verification_results = []

    for rank, piece in enumerate(retrieved_pieces, start=1):

        reference = piece["text"]

        result = classify_claim(
            premise=reference,
            hypothesis=claim
        )

        verification_record = {
            "rank": rank,
            "piece_id": piece["piece_id"],
            "page_number": piece["page_number"],
            "reference": reference,
            "similarity_score": piece["similarity_score"],
            "classification": result["classification"],
            "confidence": result["confidence"],
            "scores": result["scores"]
        }

        verification_results.append(verification_record)

        print(f"\nREFERENCE {rank}")
        print("-" * 45)

        print("Piece ID:", piece["piece_id"])
        print("Page Number:", piece["page_number"])
        print("Similarity Score:", piece["similarity_score"])
        print("Reference Evidence:", reference)

        print("\nNLI Classification:", result["classification"])
        print("NLI Confidence:", str(result["confidence"]) + "%")

        print("\nAll classification scores:")

        for label, score in result["scores"].items():
            print(f"{label}: {score}%")

    # --------------------------------------------------
    # STEP 3: VERIFICATION SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("STEP 3: VERIFICATION SUMMARY")
    print("=" * 65)

    counts = {
        "Contradiction": 0,
        "Entailment": 0,
        "Neutral": 0
    }

    for result in verification_results:
        counts[result["classification"]] += 1

    print("\nTotal references checked:", len(verification_results))
    print("Contradictions:", counts["Contradiction"])
    print("Entailments:", counts["Entailment"])
    print("Neutral:", counts["Neutral"])

    # --------------------------------------------------
    # STEP 4: EVIDENCE SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("STEP 4: EVIDENCE SUMMARY")
    print("=" * 65)

    for result in verification_results:

        print(
            f"\nReference {result['rank']} "
            f"(Page {result['page_number']})"
        )

        print("Classification:", result["classification"])
        print("Evidence:", result["reference"])

    print("\n" + "=" * 65)
    print("HALLUCINATION DETECTION PIPELINE COMPLETED!")
    print("=" * 65)

    print(
        "\nNote: Individual NLI classifications are evidence "
        "signals, not a final hallucination verdict."
    )

    return verification_results


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    sample_claim = (
        "Correlation always means that one variable "
        "causes a change in another variable."
    )

    results = detect_hallucination(
        sample_claim,
        top_k=5
    )
