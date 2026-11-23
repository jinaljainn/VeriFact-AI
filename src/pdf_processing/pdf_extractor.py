
import os
import shutil
from pathlib import Path

from pypdf import PdfReader
import pypdfium2 as pdfium
import pytesseract


# Configure Tesseract for Windows or Linux
if os.name == "nt":
    windows_tesseract = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if os.path.exists(windows_tesseract):
        pytesseract.pytesseract.tesseract_cmd = windows_tesseract
else:
    # On Streamlit Cloud, Tesseract is installed system-wide.
    tesseract_path = shutil.which("tesseract")

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


def extract_pdf_text(pdf_path):
    """
    Extract text from every PDF page.

    First attempts normal text extraction using pypdf.
    If a page contains little or no readable text,
    renders that page as an image and uses Tesseract OCR.

    Supports:
    - Text-based PDFs
    - Scanned/image-based PDFs
    - Mixed PDFs containing both types

    Page numbering starts from 1.
    """

    pdf_path = str(pdf_path)

    reader = PdfReader(pdf_path)
    extracted_pages = []

    # Open PDF for rendering scanned pages
    pdf_document = pdfium.PdfDocument(pdf_path)

    try:
        for page_index, page in enumerate(reader.pages):

            page_number = page_index + 1

            # First attempt normal text extraction
            text = page.extract_text() or ""

            # Use OCR if normal extraction returns very little text
            if len(text.strip()) < 30:

                print(
                    f"Page {page_number}: "
                    "Little or no text found. Running OCR..."
                )

                pdf_page = pdf_document[page_index]

                try:
                    # Render PDF page as an image
                    image = pdf_page.render(
                        scale=2
                    ).to_pil()

                    # Extract text from image using Tesseract
                    ocr_text = pytesseract.image_to_string(
                        image,
                        lang="eng"
                    )

                    # Prefer OCR if it extracts more text
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text

                finally:
                    pdf_page.close()

            else:
                print(
                    f"Page {page_number}: "
                    "Text extracted normally."
                )

            extracted_pages.append({
                "page_number": page_number,
                "text": text
            })

    finally:
        pdf_document.close()

    return extracted_pages


if __name__ == "__main__":

    pdf_path = "data/reference_pdfs/sample.pdf"

    pages = extract_pdf_text(pdf_path)

    print("\nTotal pages:", len(pages))

    for page in pages:
        print(f"\n--- PAGE {page['page_number']} ---")
        print(page["text"][:1000])
