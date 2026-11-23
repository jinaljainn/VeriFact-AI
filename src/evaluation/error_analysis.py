
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_RESULTS_PATH = (
    PROJECT_ROOT / "results" / "evaluation_results.csv"
)

ERROR_ANALYSIS_PATH = (
    PROJECT_ROOT / "results" / "error_analysis.csv"
)


# --------------------------------------------------
# 2. LOAD EVALUATION RESULTS
# --------------------------------------------------

def load_evaluation_results():

    if not EVALUATION_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Evaluation results not found. "
            "Run evaluation_runner.py first."
        )

    results = pd.read_csv(EVALUATION_RESULTS_PATH)

    required_columns = {
        "claim",
        "expected_verdict",
        "predicted_verdict",
        "correct"
    }

    if not required_columns.issubset(results.columns):
        raise ValueError(
            "Evaluation results are missing required columns."
        )

    return results


# --------------------------------------------------
# 3. IDENTIFY INCORRECT PREDICTIONS
# --------------------------------------------------

def identify_errors(results):

    errors = results[
        results["correct"].astype(str).str.lower() == "false"
    ].copy()

    return errors


# --------------------------------------------------
# 4. ANALYSE ERROR CATEGORIES
# --------------------------------------------------

def analyse_error_categories(errors):

    print("\nERRORS BY EXPECTED VERDICT")
    print("-" * 50)

    if errors.empty:
        print("No incorrect predictions found.")
        return

    expected_counts = (
        errors["expected_verdict"].value_counts()
    )

    print(expected_counts.to_string())

    print("\nERRORS BY PREDICTED VERDICT")
    print("-" * 50)

    predicted_counts = (
        errors["predicted_verdict"].value_counts()
    )

    print(predicted_counts.to_string())


# --------------------------------------------------
# 5. SAVE ERROR REPORT
# --------------------------------------------------

def save_error_report(errors):

    errors.to_csv(
        ERROR_ANALYSIS_PATH,
        index=False
    )

    print("\nError report saved successfully:")
    print(ERROR_ANALYSIS_PATH)


# --------------------------------------------------
# 6. DISPLAY INDIVIDUAL ERRORS
# --------------------------------------------------

def display_errors(errors):

    print("\n" + "=" * 65)
    print("             VERIFACT AI - ERROR ANALYSIS")
    print("=" * 65)

    print(f"\nTotal incorrect predictions: {len(errors)}")

    if errors.empty:
        print("\nAll predictions were correct.")
        return

    for index, (_, row) in enumerate(
        errors.iterrows(),
        start=1
    ):

        print("\n" + "-" * 65)

        print(f"ERROR {index}")

        print(f"\nClaim: {row['claim']}")

        print(
            f"Expected verdict: "
            f"{row['expected_verdict']}"
        )

        print(
            f"Predicted verdict: "
            f"{row['predicted_verdict']}"
        )

        print("\nStatus: Incorrect prediction")

    print("\n" + "=" * 65)


# --------------------------------------------------
# 7. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\nStarting VeriFact AI Error Analysis...")

    results = load_evaluation_results()

    errors = identify_errors(results)

    display_errors(errors)

    analyse_error_categories(errors)

    save_error_report(errors)

    print("\nError analysis completed!")


if __name__ == "__main__":
    main()
