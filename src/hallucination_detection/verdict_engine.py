
import sys
import re
from pathlib import Path

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hallucination_detection.hallucination_detector import (
    detect_hallucination
)

# ============================================================
# VERDICT CONFIGURATION
# ============================================================

SIMILARITY_THRESHOLD = 0.65
NLI_CONFIDENCE_THRESHOLD = 80.0
EXACT_MATCH_SIMILARITY_THRESHOLD = 0.85
MIN_SHARED_WORDS = 2

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "be", "been", "being", "that", "this", "these",
    "those", "of", "in", "on", "at", "to", "for",
    "and", "or", "as", "by", "with", "from",
    "it", "its", "their", "they", "them",
    "has", "have", "had", "does", "do",
    "according", "states", "state", "described",
    "shows", "show", "indicates", "indicated",
    "document", "solution", "there", "here",
    "which", "who"
}

NEGATION_WORDS = {
    "no", "not", "never", "without", "neither",
    "nor", "cannot", "doesn't", "isn't", "aren't",
    "wasn't", "weren't"
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = str(text or "").lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_meaningful_words(text):

    words = normalize_text(text).split()

    return {
        word for word in words
        if word not in STOP_WORDS
        and word not in NEGATION_WORDS
        and len(word) > 1
    }


def has_negation(text):

    words = set(normalize_text(text).split())
    return bool(words.intersection(NEGATION_WORDS))


# ============================================================
# DIRECT TEXTUAL EVIDENCE CHECK
# ============================================================

def detect_direct_textual_match(claim, reference):

    claim_words = get_meaningful_words(claim)
    reference_words = get_meaningful_words(reference)

    shared_words = claim_words.intersection(reference_words)

    if not claim_words or not reference_words:
        return {
            "detected": False,
            "classification": None,
            "overlap": 0.0,
            "reason": None
        }

    denominator = min(len(claim_words), len(reference_words))
    overlap = len(shared_words) / denominator

    if len(shared_words) < MIN_SHARED_WORDS or overlap < 0.80:
        return {
            "detected": False,
            "classification": None,
            "overlap": round(overlap, 3),
            "reason": None
        }

    if has_negation(claim) != has_negation(reference):
        return {
            "detected": True,
            "classification": "Contradiction",
            "overlap": round(overlap, 3),
            "reason": (
                "The claim and reference contain strongly "
                "overlapping factual terms but express "
                "opposite negation."
            )
        }

    return {
        "detected": True,
        "classification": "Entailment",
        "overlap": round(overlap, 3),
        "reason": (
            "The claim and reference share strongly "
            "overlapping factual wording with matching negation."
        )
    }


# ============================================================
# CAUSATION EQUIVALENCE CHECK
# ============================================================

def detect_causation_equivalence(claim, reference):

    claim_text = normalize_text(claim)
    reference_text = normalize_text(reference)

    equivalence_intent = bool(
        re.search(
            r"\b(?:another way to describe|same as|"
            r"equivalent to|means the same as|"
            r"is another term for)\b",
            claim_text
        )
    )

    claim_mentions_cause_effect = bool(
        re.search(r"\bcause and effect\b", claim_text)
    )

    claim_mentions_causation = bool(
        re.search(r"\bcausation\b", claim_text)
    )

    claim_is_negated = bool(
        re.search(
            r"\b(?:not|never|without|cannot|doesn't|"
            r"does not|isn't|is not)\b",
            claim_text
        )
    )

    if not (
        equivalence_intent
        and claim_mentions_cause_effect
        and claim_mentions_causation
        and not claim_is_negated
    ):
        return {
            "detected": False,
            "classification": None,
            "reason": None
        }

    reference_has_cause_effect = bool(
        re.search(r"\bcause and effect\b", reference_text)
    )

    reference_has_causal_mechanism = bool(
        re.search(
            r"\bone variable makes the other variable happen\b",
            reference_text
        )
        or re.search(
            r"\bcausation explains\b.*\bcause a difference\b",
            reference_text
        )
    )

    if (
        reference_has_cause_effect
        or reference_has_causal_mechanism
    ):
        return {
            "detected": True,
            "classification": "Entailment",
            "reason": (
                "The claim explicitly equates cause and effect "
                "with causation, and the reference explains "
                "the same causal relationship."
            )
        }

    return {
        "detected": False,
        "classification": None,
        "reason": None
    }


# ============================================================
# NUMERICAL MEDIAN VERIFICATION
# ============================================================

def detect_median_claim(claim, verification_results):
    """
    Verify a median claim using numerical facts retrieved
    from the reference document.

    The rule requires:
    1. A claim that explicitly states a median value.
    2. A reference stating the number of observations.
    3. A reference stating the value at the median position.
    """

    claim_text = normalize_text(claim)

    # Only process claims explicitly about the median.
    if not re.search(r"\bmedian\b", claim_text):
        return {
            "detected": False,
            "classification": None,
            "median_value": None,
            "claimed_value": None,
            "reason": None
        }

    # Extract the claimed median value.
    claim_match = re.search(
        r"\bmedian\b.*?\b(?:is|equals|equal to|=)\s*"
        r"(-?\d+(?:\.\d+)?)",
        claim_text
    )

    if not claim_match:
        return {
            "detected": False,
            "classification": None,
            "median_value": None,
            "claimed_value": None,
            "reason": None
        }

    claimed_value = float(claim_match.group(1))

    # Gather relevant numerical facts from retrieved references.
    observation_count = None
    ordinal_position = None
    ordinal_value = None
    dataset_values = None
    dataset_reference = None
    evidence_references = []

    for result in verification_results:

        if result.get("similarity_score", 0) < SIMILARITY_THRESHOLD:
            continue

        reference = normalize_text(result.get("reference", ""))

        # Extract number of observations.
        count_match = re.search(
            r"\bthere are\s+(\d+)\s+values?\b",
            reference
        )

        if count_match:
            observation_count = int(count_match.group(1))
            evidence_references.append(result)

        # Extract an explicit ordinal statement.
        # Example: "The 8th value in the sorted dataset is 35."
        ordinal_match = re.search(
            r"\b(\d+)(?:st|nd|rd|th)\s+value"
            r".*?\bis\s+(-?\d+(?:\.\d+)?)",
            reference
        )

        if ordinal_match:
            ordinal_position = int(ordinal_match.group(1))
            ordinal_value = float(ordinal_match.group(2))
            evidence_references.append(result)

        # Recognise an explicitly listed dataset, for example:
        # "Sort the data: 23, 27, 28, ... , 51"
        # normalize_text removes commas and colons, leaving the listed
        # dataset as whitespace-separated numbers after "sort the data".
        list_match = re.search(
            r"\bsort the data\s+"
            r"((-?\d+(?:\.\d+)?\s+){2,}"
            r"-?\d+(?:\.\d+)?)",
            reference
        )
        if list_match:
            values = [
                float(value)
                for value in list_match.group(1).split()
            ]
            if len(values) >= 3:
                dataset_values = values
                dataset_reference = result

    # We need enough numerical information to verify the median.
    if (
        observation_count is None
        or ordinal_position is None
        or ordinal_value is None
    ):
        return {
            "detected": False,
            "classification": None,
            "median_value": None,
            "claimed_value": claimed_value,
            "reason": (
                "The retrieved evidence does not contain "
                "enough numerical information to verify "
                "the median."
            )
        }

    # Calculate the median position for an odd-sized dataset.
    if observation_count % 2 == 1:

        calculated_position = (observation_count + 1) // 2

    else:
        # For even-sized datasets, this simple rule does not
        # calculate the average of the two middle values.
        return {
            "detected": False,
            "classification": None,
            "median_value": None,
            "claimed_value": claimed_value,
            "reason": (
                "The dataset has an even number of observations. "
                "The two middle values are required to calculate "
                "the median."
            )
        }

    # Prefer a complete listed dataset when its size agrees with the
    # stated observation count. This validates the dataset evidence too.
    if (
        dataset_values is not None
        and len(dataset_values) == observation_count
    ):
        sorted_values = sorted(dataset_values)
        median_value = sorted_values[calculated_position - 1]

        if (
            dataset_reference is not None
            and dataset_reference not in evidence_references
        ):
            evidence_references.append(dataset_reference)

        # If an ordinal statement is also present, it must agree with
        # the complete dataset list.
        if (
            ordinal_position == calculated_position
            and ordinal_value is not None
            and ordinal_value != median_value
        ):
            return {
                "detected": False,
                "classification": None,
                "median_value": None,
                "claimed_value": claimed_value,
                "reason": (
                    "The retrieved sorted dataset and ordinal statement "
                    "disagree about the median value."
                )
            }
    else:
        # Without a complete list, require an explicit ordinal statement.
        if ordinal_position != calculated_position or ordinal_value is None:
            return {
                "detected": False,
                "classification": None,
                "median_value": None,
                "claimed_value": claimed_value,
                "reason": (
                    "The retrieved ordinal value does not match "
                    "the calculated median position, or is unavailable."
                )
            }
        median_value = ordinal_value

    if claimed_value == median_value:
        classification = "Entailment"
        reason = (
            f"The reference states that there are "
            f"{observation_count} observations. The median "
            f"position is ({observation_count} + 1) / 2 = "
            f"{calculated_position}. The reference states that "
            f"the value at position {calculated_position} is "
            f"{median_value:g}. Therefore, the claimed median "
            f"is correct."
        )

    else:
        classification = "Contradiction"
        reason = (
            f"The reference states that there are "
            f"{observation_count} observations. The median "
            f"position is ({observation_count} + 1) / 2 = "
            f"{calculated_position}. The value at that position "
            f"is {median_value:g}, not {claimed_value:g}."
        )

    return {
        "detected": True,
        "classification": classification,
        "median_value": median_value,
        "claimed_value": claimed_value,
        "observation_count": observation_count,
        "median_position": calculated_position,
        "reason": reason,
        "evidence_references": evidence_references
    }


# ============================================================
# ANALYSIS TYPE CONFLICT CHECK
# ============================================================

def extract_analysis_type(text):

    text = str(text or "").lower()

    patterns = {
        "Univariate": r"\bunivariate\b",
        "Bivariate": r"\bbivariate\b",
        "Multivariate": r"\bmultivariate\b"
    }

    found_types = [
        label
        for label, pattern in patterns.items()
        if re.search(pattern, text)
    ]

    if len(found_types) == 1:
        return found_types[0]

    return None


def detect_category_conflict(claim, reference):

    claim_type = extract_analysis_type(claim)
    reference_type = extract_analysis_type(reference)

    if (
        claim_type is not None
        and reference_type is not None
        and claim_type != reference_type
    ):
        return {
            "detected": True,
            "claim_type": claim_type,
            "reference_type": reference_type,
            "reason": (
                f"The claim identifies the analysis as "
                f"{claim_type}, whereas the reference "
                f"identifies it as {reference_type}."
            )
        }

    return {
        "detected": False,
        "claim_type": claim_type,
        "reference_type": reference_type,
        "reason": None
    }


# ============================================================
# CORRELATION / CAUSATION EVIDENCE RELEVANCE
# ============================================================

def detect_correlation_causation_scope(claim, reference):

    claim_text = normalize_text(claim)
    reference_text = normalize_text(reference)

    claim_correlation = bool(
        re.search(r"\bcorrelation\b", claim_text)
    )

    claim_causation = bool(
        re.search(
            r"\bcausation\b|\bcausal\b|\bcause\b|\bcauses\b",
            claim_text
        )
    )

    reference_correlation = bool(
        re.search(r"\bcorrelation\b", reference_text)
    )

    reference_causation = bool(
        re.search(
            r"\bcausation\b|\bcausal\b|\bcause\b|\bcauses\b",
            reference_text
        )
    )

    causal_denial = bool(
        re.search(
            r"\b(?:does not|doesn't|do not|don't|cannot|never)"
            r"\s+(?:imply|mean|prove|establish)\b",
            reference_text
        )
        and re.search(
            r"\b(?:affect|affects|cause|causes|causal)\b",
            reference_text
        )
    )

    is_question_prompt = bool(
        re.search(
            r"\b(explain|discuss|describe|define)\s+"
            r"(?:the\s+)?(?:terms?|concepts?)\b",
            reference_text
        )
    )

    if is_question_prompt:
        return {
            "relevant": False,
            "reason": (
                "This reference is an instructional question, "
                "not a factual statement."
            )
        }

    if claim_correlation and claim_causation:

        if not (
            reference_correlation
            and (reference_causation or causal_denial)
        ):
            return {
                "relevant": False,
                "reason": (
                    "The claim compares correlation and causation, "
                    "but this passage does not provide evidence "
                    "about their relationship."
                )
            }

    elif claim_correlation:

        if reference_causation and not reference_correlation:
            return {
                "relevant": False,
                "reason": (
                    "The claim is about correlation, whereas "
                    "the reference discusses causation alone."
                )
            }

    elif claim_causation:

        if reference_correlation and not reference_causation:
            return {
                "relevant": False,
                "reason": (
                    "The claim is about causation, whereas "
                    "the reference discusses correlation alone."
                )
            }

    return {
        "relevant": True,
        "reason": None
    }


# ============================================================
# OUTLIER CLAIM VERIFICATION
# ============================================================

def detect_outlier_claim(claim, verification_results):
    """
    Verify simple claims about whether outliers are present.
    Uses only retrieved passages above the existing similarity threshold.
    Instructional prompts are ignored.
    """

    claim_text = normalize_text(claim)
    if not re.search(r"\boutliers?\b", claim_text):
        return {
            "detected": False,
            "classification": None,
            "reason": None,
            "evidence_references": []
        }

    claim_says_no_outliers = bool(
        re.search(
            r"\b(?:no|not|without|zero|none)\b"
            r"(?:\s+\w+){0,3}\s+\boutliers?\b",
            claim_text
        )
    )

    no_outlier_patterns = [
        r"\bno\s+(?:obvious\s+|significant\s+|potential\s+|clear\s+|major\s+)?outliers?\b",
        r"\bwithout\s+(?:any\s+)?outliers?\b",
        r"\bnot\s+(?:any\s+)?outliers?\b",
        r"\bzero\s+outliers?\b"
    ]
    positive_outlier_patterns = [
        r"\boutliers?\s+(?:are|were|is|was|have been|has been)\s+(?:identified|detected|found|observed|present)\b",
        r"\b(?:identified|detected|found|observed)\s+(?:\w+\s+){0,2}outliers?\b",
        r"\b(?:there\s+are|there\s+were|the\s+dataset\s+contains|the\s+data\s+contain)\s+(?:some\s+|several\s+|multiple\s+|two\s+|three\s+|four\s+|five\s+)?outliers?\b",
        r"\boutliers?\s+(?:exist|occur|appear)\b"
    ]

    evidence_references = []
    for result in verification_results:
        if result.get("similarity_score", 0) < SIMILARITY_THRESHOLD:
            continue

        reference_text = normalize_text(result.get("reference", ""))

        # A question or instruction is not evidence that outliers exist.
        if re.search(
            r"\b(?:identify|find|detect|check|look for|determine whether)\b.*\boutliers?\b",
            reference_text
        ):
            continue

        says_no = any(re.search(p, reference_text) for p in no_outlier_patterns)
        says_yes = any(re.search(p, reference_text) for p in positive_outlier_patterns)

        if not says_no and not says_yes:
            continue

        # Reference polarity: no-outlier statement = entailment of a
        # "no outliers" claim; positive statement = contradiction of it.
        if claim_says_no_outliers:
            classification = "Entailment" if says_no else "Contradiction"
        else:
            classification = "Contradiction" if says_no else "Entailment"

        evidence_references.append({
            **result,
            "outlier_classification": classification,
            "outlier_reason": (
                "The reference explicitly states that no outliers were identified."
                if says_no
                else "The reference explicitly states that outliers were identified or are present."
            )
        })

    if not evidence_references:
        return {
            "detected": False,
            "classification": None,
            "reason": None,
            "evidence_references": []
        }

    classifications = {item["outlier_classification"] for item in evidence_references}
    if classifications == {"Entailment"}:
        classification = "Entailment"
        reason = "The retrieved reference evidence agrees with the claim about outlier presence."
    elif classifications == {"Contradiction"}:
        classification = "Contradiction"
        reason = "The retrieved reference evidence conflicts with the claim about outlier presence."
    else:
        classification = "Mixed"
        reason = "The retrieved reference passages contain both supporting and contradicting statements about outliers."

    return {
        "detected": True,
        "classification": classification,
        "reason": reason,
        "evidence_references": evidence_references
    }


# ============================================================
# TIME-SERIES TREND CLAIM VERIFICATION
# ============================================================

def detect_trend_claim(claim, verification_results):
    """
    Verify direct claims about whether a time series has a consistent
    upward or downward trend.

    A retrieved passage must explicitly describe the trend. A question
    or instruction asking the reader to analyse a plot is not treated
    as factual evidence.
    """

    claim_text = normalize_text(claim)

    if not re.search(r"\btrend\b", claim_text):
        return {
            "detected": False,
            "classification": None,
            "reason": None,
            "evidence_references": []
        }

    claim_says_no_consistent_trend = bool(
        re.search(
            r"\bno\s+(?:clear\s+|consistent\s+|significant\s+|"
            r"overall\s+)?trend\b",
            claim_text
        )
        or re.search(
            r"\bdoes not\s+(?:show|have|display)\s+"
            r"(?:a\s+)?(?:clear\s+|consistent\s+|significant\s+)?trend\b",
            claim_text
        )
        or re.search(
            r"\b(?:there is|there was)\s+no\s+"
            r"(?:clear\s+|consistent\s+|significant\s+)?trend\b",
            claim_text
        )
        # Also recognise claims that spell out both trend directions:
        # "no consistent upward or downward trend".
        or re.search(
            r"\bno\s+(?:clear\s+|consistent\s+|significant\s+|overall\s+)?"
            r"(?:upward\s+or\s+downward|downward\s+or\s+upward)\s+trend\b",
            claim_text
        )
    )

    no_trend_patterns = [
        r"\bthere\s+is\s+no\s+(?:clear\s+|consistent\s+|"
        r"significant\s+|overall\s+)?trend\b",
        r"\bno\s+(?:clear\s+|consistent\s+|significant\s+|"
        r"overall\s+)?trend\s+(?:is\s+)?(?:present|observed|visible|"
        r"apparent|evident)\b",
        r"\b(?:does|do)\s+not\s+(?:show|display|exhibit)\s+"
        r"(?:a\s+)?(?:clear\s+|consistent\s+|significant\s+)?trend\b",
        r"\b(?:upward|downward)\s+trend\s+(?:is\s+)?not\s+"
        r"(?:present|observed|evident)\b",
        r"\bno\s+consistent\s+trend\s*\(\s*upward\s+or\s+downward\s*\)"
    ]

    trend_present_patterns = [
        r"\b(?:there\s+is|there\s+was)\s+(?:a\s+)?"
        r"(?:clear\s+|consistent\s+|significant\s+)?"
        r"(?:upward|downward|increasing|decreasing)\s+trend\b",
        r"\b(?:clear|consistent|significant)\s+"
        r"(?:upward|downward|increasing|decreasing)\s+trend\b",
        r"\bthe\s+series\s+(?:shows|exhibits|displays)\s+"
        r"(?:a\s+)?(?:clear\s+|consistent\s+|significant\s+)?"
        r"(?:upward|downward|increasing|decreasing)\s+trend\b"
    ]

    evidence_references = []

    for result in verification_results:
        if result.get("similarity_score", 0) < SIMILARITY_THRESHOLD:
            continue

        reference_text = normalize_text(result.get("reference", ""))

        # Ignore prompts and instructions asking for time-series analysis.
        is_analysis_prompt = bool(
            re.search(
                r"\b(?:analyse|analyze|analysis|examine|interpret)\b",
                reference_text
            )
            and re.search(
                r"\b(?:below|following)\s+"
                r"(?:time\s+series|plot|visualization|graph)\b",
                reference_text
            )
            and re.search(
                r"\b(?:find|identify|describe|discuss|determine|"
                r"explain)\s+(?:its|the)\s+"
                r"(?:characteristics|trend|pattern|features)\b",
                reference_text
            )
        )

        if is_analysis_prompt:
            continue

        says_no_trend = any(
            re.search(pattern, reference_text)
            for pattern in no_trend_patterns
        )
        says_trend_present = any(
            re.search(pattern, reference_text)
            for pattern in trend_present_patterns
        )

        if not says_no_trend and not says_trend_present:
            continue

        if claim_says_no_consistent_trend:
            classification = (
                "Entailment" if says_no_trend else "Contradiction"
            )
        else:
            classification = (
                "Contradiction" if says_no_trend else "Entailment"
            )

        evidence_references.append({
            **result,
            "trend_classification": classification,
            "trend_reason": (
                "The reference explicitly states that no consistent "
                "trend is present."
                if says_no_trend
                else "The reference explicitly describes a trend in the series."
            )
        })

    if not evidence_references:
        return {
            "detected": False,
            "classification": None,
            "reason": None,
            "evidence_references": []
        }

    classifications = {
        item["trend_classification"]
        for item in evidence_references
    }

    if classifications == {"Entailment"}:
        classification = "Entailment"
        reason = (
            "The retrieved reference explicitly states that there is "
            "no consistent trend in the time series, which supports "
            "the claim."
            if claim_says_no_consistent_trend
            else "The retrieved reference explicitly describes a trend "
            "in the time series, which supports the claim."
        )
    elif classifications == {"Contradiction"}:
        classification = "Contradiction"
        reason = (
            "The retrieved reference explicitly describes a trend "
            "that conflicts with the claim."
        )
    else:
        classification = "Mixed"
        reason = (
            "The retrieved reference passages contain conflicting "
            "statements about the time-series trend."
        )

    return {
        "detected": True,
        "classification": classification,
        "reason": reason,
        "evidence_references": evidence_references
    }


# ============================================================
# NUMERICAL RANGE VERIFICATION
# ============================================================

def detect_range_claim(claim, verification_results):
    """
    Verify a stated dataset range using a retrieved complete dataset
    or an explicit max-minus-min calculation.

    The rule does not treat an instruction to calculate the range as
    factual evidence.
    """

    claim_text = normalize_text(claim)

    if not re.search(r"\brange\b", claim_text):
        return {
            "detected": False,
            "classification": None,
            "claimed_value": None,
            "calculated_range": None,
            "reason": None,
            "evidence_references": []
        }

    claim_match = re.search(
        r"\brange\b.*?\b(?:is|equals|equal to|=)\s*"
        r"(-?\d+(?:\.\d+)?)",
        claim_text
    )
    if not claim_match:
        return {
            "detected": False,
            "classification": None,
            "claimed_value": None,
            "calculated_range": None,
            "reason": None,
            "evidence_references": []
        }

    claimed_value = float(claim_match.group(1))
    dataset_values = None
    dataset_reference = None
    arithmetic_reference = None
    arithmetic_range = None

    for result in verification_results:
        if result.get("similarity_score", 0) < SIMILARITY_THRESHOLD:
            continue

        reference = normalize_text(result.get("reference", ""))

        # Ignore instructional prompts, such as "Calculate the range".
        if re.search(r"\bcalculate\b.*\brange\b", reference):
            continue

        # Extract a listed dataset from the age exercise.
        list_match = re.search(
            r"\bsort the data\s+"
            r"((-?\d+(?:\.\d+)?\s+){2,}-?\d+(?:\.\d+)?)",
            reference
        )
        if list_match:
            values = [
                float(value)
                for value in list_match.group(1).split()
            ]
            if len(values) >= 2:
                dataset_values = values
                dataset_reference = result

        # Extract an explicit calculation, e.g. "28 (from 51-23)".
        calculation_match = re.search(
            r"(-?\d+(?:\.\d+)?)\s*"
            r"\(\s*from\s+"
            r"(-?\d+(?:\.\d+)?)\s*[-−]\s*"
            r"(-?\d+(?:\.\d+)?)\s*\)",
            reference
        )
        if calculation_match:
            stated_result = float(calculation_match.group(1))
            maximum = float(calculation_match.group(2))
            minimum = float(calculation_match.group(3))
            if abs((maximum - minimum) - stated_result) < 1e-9:
                arithmetic_range = stated_result
                arithmetic_reference = result

    # Prefer a range calculated from the complete listed dataset.
    if dataset_values is not None:
        calculated_range = max(dataset_values) - min(dataset_values)
    elif arithmetic_range is not None:
        calculated_range = arithmetic_range
    else:
        return {
            "detected": False,
            "classification": None,
            "claimed_value": claimed_value,
            "calculated_range": None,
            "reason": (
                "The retrieved evidence does not contain a complete "
                "dataset or a verifiable range calculation."
            ),
            "evidence_references": []
        }

    evidence_references = []
    if dataset_reference is not None:
        evidence_references.append(dataset_reference)
    if (
        arithmetic_reference is not None
        and arithmetic_reference is not dataset_reference
    ):
        evidence_references.append(arithmetic_reference)

    if abs(claimed_value - calculated_range) < 1e-9:
        classification = "Entailment"
        reason = (
            f"The dataset minimum is {min(dataset_values):g} and its "
            f"maximum is {max(dataset_values):g}. Therefore, the range "
            f"is {max(dataset_values):g} - {min(dataset_values):g} = "
            f"{calculated_range:g}, which matches the claim."
            if dataset_values is not None
            else f"The reference calculation confirms that the range is "
                 f"{calculated_range:g}, which matches the claim."
        )
    else:
        classification = "Contradiction"
        if dataset_values is not None:
            reason = (
                f"The dataset minimum is {min(dataset_values):g} and its "
                f"maximum is {max(dataset_values):g}. Therefore, the range "
                f"is {max(dataset_values):g} - {min(dataset_values):g} = "
                f"{calculated_range:g}, not {claimed_value:g}."
            )
        else:
            reason = (
                f"The reference calculates the range as "
                f"{calculated_range:g}, not {claimed_value:g}."
            )

    return {
        "detected": True,
        "classification": classification,
        "claimed_value": claimed_value,
        "calculated_range": calculated_range,
        "reason": reason,
        "evidence_references": evidence_references
    }


# ============================================================
# FINAL VERDICT ENGINE
# ============================================================

def generate_final_verdict(claim, verification_results):

    supporting_evidence = []
    contradicting_evidence = []
    neutral_evidence = []

    exact_match_evidence = []
    rule_based_contradictions = []
    direct_textual_evidence = []
    numerical_rule_evidence = []
    filtered_evidence = []

    normalized_claim = normalize_text(claim)

    # Run median reasoning across all retrieved references.
    median_check = detect_median_claim(
        claim,
        verification_results
    )


    outlier_check = detect_outlier_claim(
        claim,
        verification_results
    )
    outlier_reference_ids = {
        (item.get("piece_id"), item.get("page_number"))
        for item in outlier_check.get("evidence_references", [])
    }

    trend_check = detect_trend_claim(
        claim,
        verification_results
    )
    trend_reference_ids = {
        (item.get("piece_id"), item.get("page_number"))
        for item in trend_check.get("evidence_references", [])
    }

    range_check = detect_range_claim(
        claim,
        verification_results
    )
    range_reference_ids = {
        (item.get("piece_id"), item.get("page_number"))
        for item in range_check.get("evidence_references", [])
    }

    median_reference_ids = {
        (
            item.get("piece_id"),
            item.get("page_number")
        )
        for item in median_check.get("evidence_references", [])
    }

    # --------------------------------------------------------
    # STEP 1: ANALYSE RETRIEVED EVIDENCE
    # --------------------------------------------------------

    for result in verification_results:

        similarity = result["similarity_score"]
        confidence = result["confidence"]
        original_classification = result["classification"]
        reference_text = result["reference"]

        if similarity < SIMILARITY_THRESHOLD:
            continue

        evidence = {
            "piece_id": result["piece_id"],
            "page_number": result["page_number"],
            "reference": reference_text,
            "similarity_score": similarity,
            "nli_confidence": confidence,
            "confidence": confidence,
            "classification": original_classification,
            "nli_classification": original_classification,
            "rule_based_conflict": False,
            "rule_based_reason": None
        }

        # Check evidence relevance.
        relevance_check = detect_correlation_causation_scope(
            claim,
            reference_text
        )

        if not relevance_check["relevant"]:

            evidence["classification"] = "Neutral"
            evidence["relevance_filter_reason"] = (
                relevance_check["reason"]
            )

            neutral_evidence.append(evidence)
            filtered_evidence.append(evidence.copy())

            continue

        # Apply numerical median reasoning when this reference
        # contributed to the median calculation.
        reference_id = (
            result.get("piece_id"),
            result.get("page_number")
        )

        if (
            median_check["detected"]
            and reference_id in median_reference_ids
        ):

            evidence["classification"] = (
                median_check["classification"]
            )

            evidence["rule_based_reason"] = median_check["reason"]
            evidence["semantic_rule"] = "Numerical median verification"
            evidence["calculated_median"] = median_check["median_value"]
            evidence["claimed_median"] = median_check["claimed_value"]

            numerical_rule_evidence.append(evidence.copy())

        elif (
            outlier_check["detected"]
            and reference_id in outlier_reference_ids
        ):

            outlier_item = next(
                item for item in outlier_check["evidence_references"]
                if (
                    item.get("piece_id"),
                    item.get("page_number")
                ) == reference_id
            )
            evidence["classification"] = outlier_item["outlier_classification"]
            evidence["rule_based_reason"] = outlier_item["outlier_reason"]
            evidence["semantic_rule"] = "Outlier presence verification"
            direct_textual_evidence.append(evidence.copy())

        elif (
            trend_check["detected"]
            and reference_id in trend_reference_ids
        ):

            trend_item = next(
                item for item in trend_check["evidence_references"]
                if (
                    item.get("piece_id"),
                    item.get("page_number")
                ) == reference_id
            )
            evidence["classification"] = trend_item["trend_classification"]
            evidence["rule_based_reason"] = trend_item["trend_reason"]
            evidence["semantic_rule"] = "Time-series trend verification"
            direct_textual_evidence.append(evidence.copy())

        elif (
            range_check["detected"]
            and reference_id in range_reference_ids
        ):

            evidence["classification"] = range_check["classification"]
            evidence["rule_based_reason"] = range_check["reason"]
            evidence["semantic_rule"] = "Numerical range verification"
            evidence["calculated_range"] = range_check["calculated_range"]
            evidence["claimed_range"] = range_check["claimed_value"]
            numerical_rule_evidence.append(evidence.copy())

        else:

            # Exact textual match.
            normalized_reference = normalize_text(reference_text)

            is_exact_match = (
                normalized_claim == normalized_reference
                and similarity >= EXACT_MATCH_SIMILARITY_THRESHOLD
            )

            # Check analysis-type conflict.
            category_check = detect_category_conflict(
                claim,
                reference_text
            )

            # Check causation equivalence.
            causation_check = detect_causation_equivalence(
                claim,
                reference_text
            )

            # Check direct textual overlap.
            textual_check = detect_direct_textual_match(
                claim,
                reference_text
            )

            if category_check["detected"]:

                evidence["classification"] = "Contradiction"
                evidence["rule_based_conflict"] = True
                evidence["rule_based_reason"] = category_check["reason"]

                rule_based_contradictions.append(evidence.copy())

            elif causation_check["detected"]:

                evidence["classification"] = "Entailment"
                evidence["rule_based_reason"] = causation_check["reason"]
                evidence["semantic_rule"] = "Causation equivalence"

                direct_textual_evidence.append(evidence.copy())

            elif textual_check["detected"]:

                evidence["classification"] = textual_check["classification"]
                evidence["textual_overlap"] = textual_check["overlap"]
                evidence["rule_based_reason"] = textual_check["reason"]

                direct_textual_evidence.append(evidence.copy())

            if is_exact_match:
                exact_match_evidence.append(evidence.copy())

        if evidence["classification"] == "Entailment":
            supporting_evidence.append(evidence.copy())

        elif evidence["classification"] == "Contradiction":
            contradicting_evidence.append(evidence.copy())

        else:
            neutral_evidence.append(evidence.copy())

    # --------------------------------------------------------
    # STEP 2: FILTER STRONG NLI EVIDENCE
    # --------------------------------------------------------

    strong_supporting = [
        item for item in supporting_evidence
        if (
            item["nli_confidence"] >= NLI_CONFIDENCE_THRESHOLD
            and not item["rule_based_conflict"]
            and not item.get("semantic_rule")
        )
    ]

    strong_contradicting = [
        item for item in contradicting_evidence
        if (
            item["nli_confidence"] >= NLI_CONFIDENCE_THRESHOLD
            and not item["rule_based_conflict"]
            and not item.get("semantic_rule")
        )
    ]

    # --------------------------------------------------------
    # STEP 3: CHECK EXACT MATCH SUPPORT
    # --------------------------------------------------------

    strong_exact_support = [
        item for item in exact_match_evidence
        if (
            item["nli_classification"] == "Entailment"
            and item["nli_confidence"] >= NLI_CONFIDENCE_THRESHOLD
        )
    ]

    # --------------------------------------------------------
    # STEP 4: GENERATE FINAL VERDICT
    # --------------------------------------------------------

    if rule_based_contradictions:

        final_verdict = "Contradicted"
        explanation = (
            "An explicit rule-based check identified a conflict "
            "between the analysis type in the claim and the "
            "reference document."
        )

    elif median_check["detected"]:

        # Convert the internal NLI-style label to VeriFact's public verdict.
        final_verdict = (
            "Supported"
            if median_check["classification"] == "Entailment"
            else "Contradicted"
        )
        explanation = median_check["reason"]

    elif outlier_check["detected"]:

        if outlier_check["classification"] == "Entailment":
            final_verdict = "Supported"
        elif outlier_check["classification"] == "Contradiction":
            final_verdict = "Contradicted"
        else:
            final_verdict = "Unverified"

        explanation = outlier_check["reason"]

    elif trend_check["detected"]:

        if trend_check["classification"] == "Entailment":
            final_verdict = "Supported"
        elif trend_check["classification"] == "Contradiction":
            final_verdict = "Contradicted"
        else:
            final_verdict = "Unverified"

        explanation = trend_check["reason"]

    elif range_check["detected"]:

        if range_check["classification"] == "Entailment":
            final_verdict = "Supported"
        else:
            final_verdict = "Contradicted"

        explanation = range_check["reason"]

    elif direct_textual_evidence:

        direct_support = [
            item for item in direct_textual_evidence
            if item["classification"] == "Entailment"
        ]

        direct_contradiction = [
            item for item in direct_textual_evidence
            if item["classification"] == "Contradiction"
        ]

        if direct_support and not direct_contradiction:

            final_verdict = "Supported"
            explanation = (
                "Direct textual or targeted semantic evidence "
                "in the reference document supports the claim."
            )

        elif direct_contradiction and not direct_support:

            final_verdict = "Contradicted"
            explanation = (
                "Direct textual evidence expresses a "
                "negated or conflicting version of the claim."
            )

        else:

            final_verdict = "Unverified"
            explanation = (
                "The direct textual evidence contains "
                "conflicting matches."
            )

    elif strong_exact_support:

        final_verdict = "Supported"
        explanation = (
            "The claim exactly matches a statement in the "
            "reference document, and NLI identifies strong "
            "supporting evidence."
        )

    elif strong_supporting and strong_contradicting:

        final_verdict = "Unverified"
        explanation = (
            "The relevant retrieved evidence contains both "
            "supporting and contradicting signals."
        )

    elif strong_contradicting:

        final_verdict = "Contradicted"
        explanation = (
            "Strong relevant reference evidence conflicts "
            "with the claim."
        )

    elif strong_supporting:

        final_verdict = "Supported"
        explanation = (
            "Strong relevant reference evidence supports the claim."
        )

    else:

        final_verdict = "Unverified"
        explanation = (
            "No sufficiently reliable supporting or contradicting "
            "evidence was found."
        )

    # --------------------------------------------------------
    # STEP 5: PREPARE FINAL RESULT
    # --------------------------------------------------------

    return {
        "claim": claim,
        "verdict": final_verdict,
        "explanation": explanation,
        "supporting_evidence": supporting_evidence,
        "contradicting_evidence": contradicting_evidence,
        "neutral_evidence": neutral_evidence,
        "exact_match_evidence": exact_match_evidence,
        "rule_based_contradictions": rule_based_contradictions,
        "direct_textual_evidence": direct_textual_evidence,
        "numerical_rule_evidence": numerical_rule_evidence,
        "outlier_rule_evidence": [
            item for item in supporting_evidence + contradicting_evidence
            if item.get("semantic_rule") == "Outlier presence verification"
        ],
        "outlier_check": outlier_check,
        "trend_rule_evidence": [
            item for item in supporting_evidence + contradicting_evidence
            if item.get("semantic_rule") == "Time-series trend verification"
        ],
        "trend_check": trend_check,
        "range_check": range_check,
        "filtered_evidence": filtered_evidence
    }


# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

def display_final_verdict(result):

    print("\n" + "=" * 70)
    print("VERIFACT AI - FINAL VERDICT")
    print("=" * 70)

    print("\nCLAIM:")
    print(result["claim"])

    print("\nFINAL VERDICT:")
    print(result["verdict"])

    print("\nEXPLANATION:")
    print(result["explanation"])

    sections = [
        ("SUPPORTING EVIDENCE", "supporting_evidence"),
        ("CONTRADICTING EVIDENCE", "contradicting_evidence"),
        ("NEUTRAL / INCONCLUSIVE EVIDENCE", "neutral_evidence"),
        ("DIRECT TEXTUAL EVIDENCE", "direct_textual_evidence"),
        ("NUMERICAL RULE EVIDENCE", "numerical_rule_evidence"),
        ("OUTLIER RULE EVIDENCE", "outlier_rule_evidence"),
        ("TIME-SERIES TREND RULE EVIDENCE", "trend_rule_evidence"),
        ("RULE-BASED CONTRADICTION CHECKS", "rule_based_contradictions"),
        ("FILTERED IRRELEVANT EVIDENCE", "filtered_evidence"),
        ("EXACT MATCH EVIDENCE", "exact_match_evidence")
    ]

    for title, key in sections:

        print("\n" + "-" * 70)
        print(title)
        print("-" * 70)

        if result.get(key):

            for evidence in result[key]:

                print("\nPiece ID:", evidence["piece_id"])
                print("Page:", evidence["page_number"])
                print("Reference:", evidence["reference"])
                print("Similarity:", evidence["similarity_score"])
                print("Original NLI:", evidence["nli_classification"])
                print("Final Classification:", evidence["classification"])
                print("NLI Confidence:", evidence["nli_confidence"], "%")

                if evidence.get("calculated_median") is not None:
                    print(
                        "Calculated median:",
                        evidence["calculated_median"]
                    )

                if evidence.get("claimed_median") is not None:
                    print(
                        "Claimed median:",
                        evidence["claimed_median"]
                    )

                if evidence.get("textual_overlap") is not None:
                    print("Textual overlap:", evidence["textual_overlap"])

                if evidence.get("rule_based_reason"):
                    print("Rule-based reason:", evidence["rule_based_reason"])

                if evidence.get("relevance_filter_reason"):
                    print(
                        "Relevance filter:",
                        evidence["relevance_filter_reason"]
                    )

        else:
            print("No evidence found.")

    print("\n" + "=" * 70)
    print("FINAL VERDICT GENERATION COMPLETED!")
    print("=" * 70)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    sample_claim = (
        "The annual earthquake time series has no consistent upward or downward trend over the entire period."
    )

    print("\nRunning hallucination detection pipeline...")

    verification_results = detect_hallucination(
        claim=sample_claim,
        top_k=5
    )

    final_result = generate_final_verdict(
        sample_claim,
        verification_results
    )

    display_final_verdict(final_result)


