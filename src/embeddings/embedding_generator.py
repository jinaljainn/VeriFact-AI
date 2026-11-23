
import json
from pathlib import Path
from fastembed import TextEmbedding

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "test_data" / "semantic_pieces.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "test_data" / "embeddings.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def generate_embeddings_from_pieces(semantic_pieces):
    """
    Generate embeddings directly from semantic pieces.

    This function is designed for processing uploaded PDFs
    without relying on fixed JSON input files.
    """

    if not semantic_pieces:
        raise ValueError("No semantic pieces provided.")

    print("\nLoading embedding model...")

    model = TextEmbedding(model_name=MODEL_NAME)

    sentences = [
        piece["text"] for piece in semantic_pieces
    ]

    print("\nGenerating embeddings...")

    vectors = list(model.embed(sentences))

    embedding_results = []

    for piece, vector in zip(semantic_pieces, vectors):

        embedding_results.append({
            "piece_id": piece["piece_id"],
            "page_number": piece["page_number"],
            "text": piece["text"],
            "embedding": vector.tolist()
        })

    print("\nEmbedding generation completed!")

    print("Total pieces:", len(embedding_results))
    print("Embedding dimensions:", len(embedding_results[0]["embedding"]))

    return embedding_results


def generate_embeddings():

    print("\nLoading semantic pieces...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Semantic pieces file not found: {INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        semantic_pieces = json.load(file)

    if not semantic_pieces:
        print("No semantic pieces found.")
        return

    embedding_results = generate_embeddings_from_pieces(
        semantic_pieces
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            embedding_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 55)
    print("VERIFACT AI - EMBEDDING GENERATION")
    print("=" * 55)

    print("\nTotal sentences processed:", len(embedding_results))
    print("Embedding dimensions:", len(embedding_results[0]["embedding"]))
    print("Model used:", MODEL_NAME)

    print("\nSample embedding:")
    print("Piece ID:", embedding_results[0]["piece_id"])
    print("Page:", embedding_results[0]["page_number"])
    print("Text:", embedding_results[0]["text"])
    print("First 5 vector values:", embedding_results[0]["embedding"][:5])

    print("\nEmbedding generation completed successfully!")
    print("Saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    generate_embeddings()
