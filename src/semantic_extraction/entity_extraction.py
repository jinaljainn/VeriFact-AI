
import json
import re
import spacy
from pathlib import Path

# --------------------------------------------------
# STEP 1: PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "test_data" / "semantic_pieces.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "test_data" / "entities.json"
)


# --------------------------------------------------
# STEP 2: LOAD SPACY MODEL
# --------------------------------------------------

print("\nLoading spaCy model...")

nlp = spacy.load("en_core_web_sm")

print("spaCy model loaded successfully!")


# --------------------------------------------------
# STEP 3: FALSE POSITIVE FILTER
# --------------------------------------------------

FALSE_POSITIVES = {
    "display", "analyze", "analyse", "calculate",
    "identify", "discuss", "format", "choose",
    "implement", "determine", "explain", "show",
    "find", "write", "create", "print",
    "iqr", "q1", "q2", "q3", "dataframe",
    "pandas", "numpy", "csv", "soln"
}


# --------------------------------------------------
# STEP 4: ENTITY CATEGORY MAPPING
# --------------------------------------------------

def map_entity_category(entity_type):

    mapping = {
        "PERSON": "PERSONAL_INFORMATION",
        "ORG": "PERSONAL_INFORMATION",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "FAC": "LOCATION",
        "DATE": "TIME",
        "TIME": "TIME",
        "MONEY": "NUMBER",
        "CARDINAL": "NUMBER",
        "ORDINAL": "NUMBER",
        "QUANTITY": "NUMBER",
        "PERCENT": "NUMBER"
    }

    return mapping.get(entity_type, "OTHER")


# --------------------------------------------------
# STEP 5: REGEX-BASED EXTRACTION
# --------------------------------------------------

def extract_dates_and_numbers(text):

    entities = []

    # Date formats such as 10-12-2024, 2024/01/01
    date_pattern = (
        r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
        r"|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"
    )

    for match in re.finditer(date_pattern, text):

        entities.append({
            "entity_text": match.group(),
            "entity_type": "TIME",
            "start": match.start(),
            "end": match.end()
        })

    # Numeric values, percentages and decimal numbers
    number_pattern = (
        r"(?<![\w.])"
        r"\d+(?:,\d{3})*(?:\.\d+)?"
        r"\s*%?"
        r"(?![\w.])"
    )

    for match in re.finditer(number_pattern, text):

        # Do not extract numbers already included in dates
        overlaps_date = any(
            match.start() < entity["end"]
            and match.end() > entity["start"]
            for entity in entities
        )

        if not overlaps_date:

            entities.append({
                "entity_text": match.group().strip(),
                "entity_type": "NUMBER",
                "start": match.start(),
                "end": match.end()
            })

    return entities


# --------------------------------------------------
# STEP 6: EXTRACT ENTITIES FROM EACH PIECE
# --------------------------------------------------

def extract_entities(semantic_pieces):

    extracted_entities = []

    for piece in semantic_pieces:

        text = piece["text"]

        doc = nlp(text)

        candidates = []

        # Extract entities identified by spaCy
        for entity in doc.ents:

            entity_text = entity.text.strip()

            if entity_text.lower() in FALSE_POSITIVES:
                continue

            category = map_entity_category(entity.label_)

            if category == "OTHER":
                continue

            # Avoid duplicate date and number extraction
            if category in ["TIME", "NUMBER"]:
                continue

            candidates.append({
                "entity_text": entity_text,
                "entity_type": category,
                "original_label": entity.label_,
                "start": entity.start_char,
                "end": entity.end_char
            })

        # Extract dates and numbers using regular expressions
        regex_entities = extract_dates_and_numbers(text)

        for entity in regex_entities:

            entity["original_label"] = "REGEX"

            candidates.append(entity)

        # Sort entities according to their position in the sentence
        candidates.sort(key=lambda x: x["start"])

        # Save entities with their original source information
        for entity in candidates:

            extracted_entities.append({

                "entity_id": len(extracted_entities) + 1,

                "piece_id": piece["piece_id"],

                "page_number": piece["page_number"],

                "entity_text": entity["entity_text"],

                "entity_type": entity["entity_type"],

                "original_label": entity["original_label"]

            })

    return extracted_entities


# --------------------------------------------------
# STEP 7: MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    print("\nReading semantic pieces...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        semantic_pieces = json.load(file)

    print(f"Total semantic pieces: {len(semantic_pieces)}")

    print("\nExtracting entities...")

    entities = extract_entities(semantic_pieces)


    # --------------------------------------------------
    # STEP 8: DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("VERIFACT AI - IMPROVED ENTITY EXTRACTION")
    print("=" * 60)

    print(f"\nTotal entities extracted: {len(entities)}\n")

    for entity in entities:

        print("-" * 60)

        print(f"Entity ID: {entity['entity_id']}")
        print(f"Piece ID: {entity['piece_id']}")
        print(f"Page Number: {entity['page_number']}")
        print(f"Entity: {entity['entity_text']}")
        print(f"Category: {entity['entity_type']}")
        print(f"Original Label: {entity['original_label']}")


    # --------------------------------------------------
    # STEP 9: SAVE RESULTS TO JSON
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

        json.dump(
            entities,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("ENTITY EXTRACTION COMPLETED!")
    print("=" * 60)

    print(f"Total entities saved: {len(entities)}")
    print(f"Output file: {OUTPUT_PATH}")
