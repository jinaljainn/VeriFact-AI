
import re
import sys
import json
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from pdf_processing.pdf_extractor import extract_pdf_text

INPUT_PDF = PROJECT_ROOT / "data" / "reference_pdfs" / "sample.pdf"

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "test_data" / "semantic_pieces.json"
)


# --------------------------------------------------
# STEP 1: CLEAN LINES
# --------------------------------------------------

def clean_line(line):

    line = line.strip()
    line = re.sub(r'\s+', ' ', line)

    line = re.sub(r'^[●•▪◦]+\s*', '', line)

    line = re.sub(
        r'^(Soln|Solution|Answer)\s*[:;]\s*',
        '',
        line,
        flags=re.IGNORECASE
    )

    return line.strip()


# --------------------------------------------------
# STEP 2: REMOVE HEADERS AND IRRELEVANT LINES
# --------------------------------------------------

def is_irrelevant_line(line):

    patterns = [
        r'^Question\s+Marks$',
        r'^Computer Science\s*&\s*Engineering$',
        r'^B\.M\.S COLLEGE OF ENGINEERING',
        r'^\(Autonomous Institute',
        r'^PART[- ]?[AB]\b',
        r'^Course Code:',
        r'^Semester:',
        r'^Maximum Marks:',
        r'^Date:',
        r'^Internals[- ]',
        r'^No\.$',
        r'^Marks$',
        r'^Total\s+\d+\s+Marks',
        r'^Each carries \d+M',
        r'^x\d+\s*=\s*\d+\s*Marks$'
    ]

    for pattern in patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True

    if re.fullmatch(r'[\d\s.,:/-]+', line):
        return True

    return False


# --------------------------------------------------
# STEP 3: IDENTIFY MARKING SCHEME LINES
# --------------------------------------------------

def is_marking_line(line):

    patterns = [
        r'^Definition of .*\b\d+\s*M\b',
        r'^Significance\s*[-–:]\s*\d+\s*M',
        r'^Python demo\s*[-–:]\s*\d+\s*M',
        r'^Difference from .*[-–:]\s*\d+\s*M',
        r'^\d+\s*M\s*(and|&|\+)',
        r'^Each point carries',
        r'^Choice between question'
    ]

    return any(
        re.search(pattern, line, re.IGNORECASE)
        for pattern in patterns
    )


# --------------------------------------------------
# STEP 4: IDENTIFY QUESTION BOUNDARIES
# --------------------------------------------------

def is_question_start(line):

    patterns = [
        r'^\d+\s*[a-z]\s+',       # 2a, 3b, 4a
        r'^\d+\s+[A-Z]',          # 1 Explain...
        r'^\d+\s*[a-z]\s*[ivx]+\)',  # 2b i)
        r'^[a-e]\.\s+',           # a. b. c.
        r'^[ivx]+\)\s+',          # i) ii)
    ]

    return any(
        re.match(pattern, line, re.IGNORECASE)
        for pattern in patterns
    )


# --------------------------------------------------
# STEP 5: IDENTIFY CODE
# --------------------------------------------------

def is_code_line(line):

    patterns = [
        r'^import\s+',
        r'^from\s+\w+\s+import',
        r'^print\s*\(',
        r'^df\s*=',
        r'^for\s+',
        r'^if\s+',
        r'^return\s+',
        r'^#',
        r'^[\w]+\s*=\s*pd\.',
        r'^[\w]+\s*=\s*np\.'
    ]

    return any(
        re.search(pattern, line)
        for pattern in patterns
    )


# --------------------------------------------------
# STEP 6: RECOGNISE HEADINGS
# --------------------------------------------------

def is_heading(line):

    patterns = [
        r'^Types of Skewness',
        r'^Positive Skewness',
        r'^Negative Skewness',
        r'^Zero Skewness',
        r'^Impact of ',
        r'^Interpretation of ',
        r'^Differences between ',
        r'^Sample output$',
        r'^Sample dataset$'
    ]

    return any(
        re.search(pattern, line, re.IGNORECASE)
        for pattern in patterns
    )


# --------------------------------------------------
# STEP 7: RECONSTRUCT PARAGRAPHS
# --------------------------------------------------

def reconstruct_paragraphs(page_text):

    lines = [
        clean_line(line)
        for line in page_text.splitlines()
    ]

    paragraphs = []
    current = ""

    def save_current():

        nonlocal current

        if current.strip():
            paragraphs.append(current.strip())

        current = ""

    for line in lines:

        if not line:
            continue

        if is_irrelevant_line(line):
            continue

        if is_marking_line(line):
            continue

        # Start a new piece at a question boundary.
        if is_question_start(line):
            save_current()
            current = line
            continue

        # Keep code separate.
        if is_code_line(line):
            save_current()
            paragraphs.append(line)
            continue

        # Keep headings separate.
        if is_heading(line):
            save_current()
            paragraphs.append(line)
            continue

        if not current:
            current = line
            continue

        previous_ends_sentence = bool(
            re.search(r'[.!?]["\')\]]?$', current)
        )

        ends_with_hyphen = current.endswith("-")

        if ends_with_hyphen:
            current = current[:-1] + line

        elif previous_ends_sentence:
            save_current()
            current = line

        else:
            current += " " + line

    save_current()

    return paragraphs


# --------------------------------------------------
# STEP 8: SPLIT PARAGRAPHS INTO SEMANTIC CHUNKS
# --------------------------------------------------

def split_into_chunks(paragraph, max_sentences=3):

    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9])',
        paragraph
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) >= 12
    ]

    chunks = []

    for i in range(0, len(sentences), max_sentences):

        chunk = " ".join(
            sentences[i:i + max_sentences]
        )

        if chunk:
            chunks.append(chunk)

    return chunks


# --------------------------------------------------
# STEP 9: EXTRACT SEMANTIC PIECES
# --------------------------------------------------

def extract_semantic_pieces(extracted_pages):

    semantic_pieces = []

    for page in extracted_pages:

        page_number = page["page_number"]

        paragraphs = reconstruct_paragraphs(
            page["text"]
        )

        for paragraph in paragraphs:

            chunks = split_into_chunks(paragraph)

            for chunk in chunks:

                semantic_pieces.append({
                    "piece_id": len(semantic_pieces) + 1,
                    "page_number": page_number,
                    "text": chunk
                })

    return semantic_pieces


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    print("\nReading reference PDF...\n")

    pages = extract_pdf_text(str(INPUT_PDF))

    print("Reconstructing paragraphs and detecting boundaries...\n")

    semantic_pieces = extract_semantic_pieces(pages)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            semantic_pieces,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("=" * 60)
    print("VERIFACT AI - SEMANTIC EXTRACTION")
    print("=" * 60)

    print("\nTotal semantic pieces:", len(semantic_pieces))

    print("\nFirst 30 extracted pieces:\n")

    for piece in semantic_pieces[:30]:

        print("-" * 60)
        print("Piece ID:", piece["piece_id"])
        print("Page:", piece["page_number"])
        print("Text:", piece["text"])

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETED SUCCESSFULLY!")
    print("Total pieces saved:", len(semantic_pieces))
    print("Output file:", OUTPUT_FILE)
    print("=" * 60)
