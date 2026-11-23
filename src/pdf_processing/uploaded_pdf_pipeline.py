
import sys
import tempfile
from pathlib import Path

# Locate the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_processing.pdf_extractor import extract_pdf_text
from src.semantic_extraction.semantic_extraction import extract_semantic_pieces
from src.embeddings.embedding_generator import generate_embeddings_from_pieces


def process_uploaded_pdf(pdf_bytes):
    """
    Process an uploaded PDF entirely in memory, apart from a temporary
    file needed by the PDF extractor.

    Returns extracted pages, semantic pieces, and embeddings.
    """

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    temp_path = None

    try:
        # 1. Temporarily save the uploaded PDF
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = Path(temp_file.name)

        print("\nStep 1: Extracting PDF text...")

        # 2. Extract text page by page
        pages = extract_pdf_text(str(temp_path))

        if not pages or not any(page["text"].strip() for page in pages):
            raise ValueError(
                "No readable text was found in the uploaded PDF."
            )

        print("Total pages extracted:", len(pages))

        # 3. Extract semantic pieces
        print("\nStep 2: Extracting semantic pieces...")

        semantic_pieces = extract_semantic_pieces(pages)

        if not semantic_pieces:
            raise ValueError("No semantic pieces could be extracted.")

        print("Total semantic pieces:", len(semantic_pieces))

        # 4. Generate embeddings
        print("\nStep 3: Generating embeddings...")

        embeddings = generate_embeddings_from_pieces(semantic_pieces)

        if not embeddings:
            raise ValueError("Embedding generation failed.")

        print("Total embeddings:", len(embeddings))
        print(
            "Embedding dimensions:",
            len(embeddings[0]["embedding"])
        )

        # 5. Return everything without overwriting saved sample data
        return {
            "pages": pages,
            "semantic_pieces": semantic_pieces,
            "embeddings": embeddings,
            "total_pages": len(pages),
            "total_pieces": len(semantic_pieces)
        }

    finally:
        # 6. Delete the temporary PDF
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":

    print("=" * 60)
    print("VERIFACT AI - UPLOADED PDF PIPELINE TEST")
    print("=" * 60)

    sample_pdf = (
        PROJECT_ROOT
        / "data"
        / "reference_pdfs"
        / "sample.pdf"
    )

    if not sample_pdf.exists():
        raise FileNotFoundError(
            f"Sample PDF not found: {sample_pdf}"
        )

    with open(sample_pdf, "rb") as file:
        pdf_bytes = file.read()

    result = process_uploaded_pdf(pdf_bytes)

    print("\n" + "=" * 60)
    print("PIPELINE TEST SUMMARY")
    print("=" * 60)

    print("Total pages:", result["total_pages"])
    print("Total semantic pieces:", result["total_pieces"])
    print("Total embeddings:", len(result["embeddings"]))

    print("\nUploaded PDF pipeline completed successfully!")
