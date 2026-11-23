
import json
import numpy as np
from pathlib import Path
from fastembed import TextEmbedding

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_FILE = (
    PROJECT_ROOT / "data" / "test_data" / "embeddings.json"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

def load_embedding_model():

    return TextEmbedding(model_name=MODEL_NAME)


# --------------------------------------------------
# LOAD SAVED REFERENCE EMBEDDINGS
# --------------------------------------------------

def load_reference_embeddings():

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_FILE}"
        )

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as file:
        reference_data = json.load(file)

    if not reference_data:
        raise ValueError("No reference embeddings found.")

    texts = [item["text"] for item in reference_data]

    embeddings = np.array(
        [item["embedding"] for item in reference_data],
        dtype=np.float32
    )

    return reference_data, texts, embeddings


# --------------------------------------------------
# RETRIEVE FROM PROVIDED EMBEDDINGS
# --------------------------------------------------

def retrieve_similar_pieces_from_embeddings(
    query,
    reference_embeddings,
    top_k=5,
    model=None
):
    """
    Retrieve relevant reference pieces from embeddings
    supplied directly by the caller.

    This function supports uploaded PDF processing.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not reference_embeddings:
        raise ValueError("No reference embeddings provided.")

    if model is None:
        model = load_embedding_model()

    # Prepare query for BGE retrieval
    query_text = (
        "Represent this sentence for searching relevant passages: "
        + query
    )

    # Generate query embedding
    query_vector = np.array(
        list(model.embed([query_text]))[0],
        dtype=np.float32
    )

    # Convert reference embeddings into NumPy array
    reference_vectors = np.array(
        [
            item["embedding"]
            for item in reference_embeddings
        ],
        dtype=np.float32
    )

    # Calculate cosine similarity
    query_norm = np.linalg.norm(query_vector)

    reference_norms = np.linalg.norm(
        reference_vectors,
        axis=1
    )

    similarities = np.dot(
        reference_vectors,
        query_vector
    ) / (reference_norms * query_norm + 1e-10)

    # Select top matching pieces
    actual_top_k = min(top_k, len(reference_embeddings))

    top_indices = np.argsort(similarities)[::-1][:actual_top_k]

    results = []

    for index in top_indices:

        result = {
            "piece_id": reference_embeddings[index]["piece_id"],
            "page_number": reference_embeddings[index]["page_number"],
            "text": reference_embeddings[index]["text"],
            "similarity_score": round(
                float(similarities[index]), 4
            )
        }

        results.append(result)

    return results


# --------------------------------------------------
# ORIGINAL FILE-BASED RETRIEVAL
# --------------------------------------------------

def retrieve_similar_pieces(query, top_k=5):

    print("\nLoading embedding model...")

    model = load_embedding_model()

    reference_data, texts, reference_embeddings = (
        load_reference_embeddings()
    )

    return retrieve_similar_pieces_from_embeddings(
        query=query,
        reference_embeddings=reference_data,
        top_k=top_k,
        model=model
    )


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("VERIFACT AI - SEMANTIC RETRIEVAL")
    print("=" * 60)

    query = (
        "Correlation always means that one variable "
        "causes a change in another variable."
    )

    print("\nInput claim:")
    print(query)

    print("\nRetrieving top 5 similar reference pieces...")

    retrieved_results = retrieve_similar_pieces(
        query,
        top_k=5
    )

    print("\n" + "=" * 60)
    print("TOP 5 RETRIEVED REFERENCE PIECES")
    print("=" * 60)

    for rank, result in enumerate(retrieved_results, start=1):

        print(f"\nRank: {rank}")
        print("Piece ID:", result["piece_id"])
        print("Page Number:", result["page_number"])
        print("Similarity Score:", result["similarity_score"])
        print("Reference Text:", result["text"])

    print("\n" + "=" * 60)
    print("SEMANTIC RETRIEVAL COMPLETED!")
    print("=" * 60)
