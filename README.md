# VeriFact AI
## LLM Hallucination Detection and Correction Using Reference Documents

### 1. Project Overview

VeriFact AI is a document-grounded hallucination detection and correction system. It checks whether a user-provided claim is supported by an uploaded PDF reference document.

The system classifies claims into three categories:

- **Supported**
- **Contradicted**
- **Unverified**

When a claim is contradicted, the system provides a correction based on relevant information retrieved from the reference document.

VeriFact AI focuses on evidence-based verification rather than independently determining universal factual truth.

### 2. Main Features

- PDF document upload
- Text extraction from text-based PDFs
- OCR-based extraction from scanned PDFs
- Automatic page-by-page OCR fallback
- Support for mixed PDFs containing text and scanned pages
- Semantic text segmentation
- Named entity extraction
- Text embeddings and semantic retrieval
- Natural Language Inference (NLI)
- Rule-based verification
- Three-category verdict generation
- Evidence-based correction
- Evidence report generation
- Downloadable PDF reports
- Evaluation using a labelled dataset

### 3. Technology Stack

- Python 3.11
- Streamlit
- FastEmbed
- ONNX Runtime
- Tokenizers
- spaCy
- Scikit-learn
- Pandas
- NumPy
- PyPDF
- pypdfium2
- pytesseract
- Tesseract OCR
- ReportLab

### 4. Project Structure

```text
VeriFact_AI/
|
|-- data/
|   |-- evaluation/
|   |-- reference_pdfs/
|   |-- test_data/
|
|-- frontend/
|   |-- app.py
|
|-- results/
|   |-- evaluation_results.csv
|   |-- evaluation_summary.json
|   |-- evidence_report.json
|   |-- evidence_report.pdf
|
|-- src/
|   |-- embeddings/
|   |-- evaluation/
|   |-- hallucination_detection/
|   |-- nli/
|   |-- pdf_processing/
|   |-- semantic_extraction/
|
|-- .gitignore
|-- packages.txt
|-- requirements.txt
|-- README.md
```

### 5. Installation and Local Setup

**Prerequisites:**

- Python 3.11
- Tesseract OCR
- Windows or Linux operating system

**Step 1: Clone the repository**

```powershell
git clone https://github.com/jinaljainn/VeriFact-AI.git
```

Navigate into the project directory:

```powershell
cd VeriFact-AI
```

**Step 2: Create a virtual environment**

```powershell
python -m venv .venv
```

**Step 3: Activate the virtual environment**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Step 4: Install Python dependencies**

```powershell
python -m pip install -r requirements.txt
```

**Step 5: Install Tesseract OCR on Windows**

Tesseract is a separate application and must be installed independently of Python packages.

Download the Windows installer from:

https://github.com/UB-Mannheim/tesseract/wiki

During installation:

- Select the English language data.
- Complete the installation.
- Use the default installation directory if possible:

```text
C:\Program Files\Tesseract-OCR
```

The application checks this path automatically on Windows.

Verify the installation in PowerShell:

```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

On Linux, install Tesseract using your operating system's package manager. For Debian or Ubuntu:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### 6. Running the Application Locally

From the project root directory, execute:

```powershell
streamlit run frontend/app.py
```

Open the local URL displayed in the terminal.

1. Upload a PDF document.
2. Enter a claim to verify.
3. Run the verification process.
4. Review the verdict and supporting evidence.
5. Download the generated evidence report.

### 7. Deploying on Streamlit Community Cloud

VeriFact AI can be deployed using Streamlit Community Cloud.

**Deployment steps:**

1. Push the project files to a GitHub repository.
2. Visit https://share.streamlit.io/
3. Sign in using your GitHub account.
4. Select **Create app**.
5. Choose the `VeriFact-AI` repository.
6. Select the `main` branch.
7. Set the main file path to:

```text
frontend/app.py
```

8. Click **Deploy**.

The root-level `requirements.txt` contains the Python dependencies. The `packages.txt` file specifies the system package required for OCR:

```text
tesseract-ocr
```

This allows Streamlit Community Cloud to install Tesseract for Linux-based deployment.

### 8. PDF Processing and OCR

VeriFact AI supports different types of PDF documents.

**Text-based PDFs:**
- Text is extracted using PyPDF.

**Scanned PDFs:**
- PDF pages are converted into images using pypdfium2.
- Tesseract OCR recognizes the text from those images.

**Mixed PDFs:**
- Each page is processed independently.
- Normal text extraction is attempted first.
- OCR is used when a page contains little or no extractable text.

The extracted text is then passed to semantic extraction and embedding generation.

OCR accuracy depends on scan quality, document layout, and available language data. Poor-quality scans and handwritten documents may not be recognized accurately.

### 9. Evaluation

The project includes a labelled evaluation dataset containing 45 claims:

- 15 Supported
- 15 Contradicted
- 15 Unverified

The recorded evaluation run achieved:

| Metric | Result |
|---|---:|
| Total claims | 45 |
| Correct predictions | 45 |
| Incorrect predictions | 0 |
| Accuracy | 100% |

The result is based on the project's current evaluation dataset. It does not establish 100% accuracy on unseen documents or claims.

### 10. Output

The system generates:

- Final verification verdict
- Supporting and contradicting evidence
- Relevant reference passages
- Corrected claim when applicable
- Evidence report in JSON format
- Downloadable PDF report

### 11. Important Limitations

- VeriFact AI verifies claims against the supplied reference document. It does not independently establish universal factual truth.
- A claim classified as Unverified means sufficient evidence was not found in the supplied document.
- OCR may produce recognition errors, particularly with poor-quality scans, handwriting, unusual layouts, or unsupported languages.
- Verification quality depends on the quality and relevance of the uploaded reference document.
- Evaluation results on a limited dataset may not represent performance on all types of documents and claims.

### 12. Project Purpose

The goal of VeriFact AI is to improve the reliability and transparency of AI-generated information by grounding claim verification and corrections in reference documents.

---

**Developed as an academic mini-project.**