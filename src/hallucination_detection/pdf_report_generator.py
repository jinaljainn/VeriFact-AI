
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from xml.sax.saxutils import escape


# ------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

JSON_FILE = PROJECT_ROOT / "results" / "evidence_report.json"
PDF_FILE = PROJECT_ROOT / "results" / "evidence_report.pdf"


# ------------------------------------------------
# SAFE TEXT FORMATTING
# ------------------------------------------------

def safe_text(value):
    """Safely format text for PDF paragraphs."""
    return escape(str(value if value is not None else "Not available"))


# ------------------------------------------------
# MAIN PDF REPORT GENERATOR
# ------------------------------------------------

def create_pdf_report(report=None, output_path=None):
    """
    Generate a PDF verification report.

    Parameters:
        report: Dictionary containing verification results.
                If None, loads the existing evidence_report.json.

        output_path: Location where the PDF should be saved.
                     If None, uses the default results folder.

    Returns:
        Path of the generated PDF.
    """

    print("\nLoading evidence report...")

    # Load saved JSON if report data is not provided
    if report is None:

        if not JSON_FILE.exists():
            raise FileNotFoundError(
                f"Evidence report not found: {JSON_FILE}"
            )

        with open(JSON_FILE, "r", encoding="utf-8") as file:
            report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError("Report must be provided as a dictionary.")

    print("Evidence report loaded successfully!")

    # Select output location
    if output_path is None:
        output_path = PDF_FILE

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # PDF DOCUMENT CONFIGURATION
    # ------------------------------------------------

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()

    # ------------------------------------------------
    # STYLES
    # ------------------------------------------------

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=23,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14,
        spaceAfter=7
    )

    small_style = ParagraphStyle(
        "SmallText",
        parent=body_style,
        fontSize=8,
        leading=11
    )

    story = []

    # ------------------------------------------------
    # TITLE
    # ------------------------------------------------

    story.append(
        Paragraph("VERIFACT AI", title_style)
    )

    story.append(
        Paragraph(
            "LLM Hallucination Detection and Correction Report",
            subtitle_style
        )
    )

    story.append(Spacer(1, 10))

    # ------------------------------------------------
    # REPORT INFORMATION
    # ------------------------------------------------

    story.append(
        Paragraph("1. REPORT INFORMATION", heading_style)
    )

    report_data = [
        ["Project", safe_text(report.get("project", "VeriFact AI"))],
        [
            "Reference Document",
            safe_text(report.get("reference_document"))
        ],
        [
            "Generated At",
            safe_text(report.get("report_generated_at"))
        ]
    ]

    info_table = Table(
        report_data,
        colWidths=[4 * cm, 12 * cm]
    )

    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1),
         colors.HexColor("#DCE6F1")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")
    ]))

    story.append(info_table)

    # ------------------------------------------------
    # ORIGINAL CLAIM
    # ------------------------------------------------

    story.append(
        Paragraph("2. ORIGINAL CLAIM", heading_style)
    )

    original_claim = safe_text(
        report.get("original_claim")
    )

    claim_table = Table(
        [[Paragraph(original_claim, body_style)]],
        colWidths=[16 * cm]
    )

    claim_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1),
         colors.HexColor("#F2F2F2")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 10)
    ]))

    story.append(claim_table)

    # ------------------------------------------------
    # FINAL VERDICT
    # ------------------------------------------------

    story.append(
        Paragraph("3. FINAL VERDICT", heading_style)
    )

    verdict_data = report.get("final_verdict", {})

    verdict = verdict_data.get("status", "Unverified")

    verdict_colors = {
        "Supported": colors.HexColor("#D9EAD3"),
        "Contradicted": colors.HexColor("#F4CCCC"),
        "Unverified": colors.HexColor("#FFF2CC")
    }

    verdict_background = verdict_colors.get(
        verdict,
        colors.HexColor("#EEEEEE")
    )

    verdict_table = Table(
        [[Paragraph(
            f"<b>{safe_text(verdict).upper()}</b>",
            ParagraphStyle(
                "VerdictText",
                parent=body_style,
                alignment=TA_CENTER,
                fontSize=14
            )
        )]],
        colWidths=[16 * cm]
    )

    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), verdict_background),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 12)
    ]))

    story.append(verdict_table)

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            safe_text(verdict_data.get("explanation")),
            body_style
        )
    )

    # ------------------------------------------------
    # CORRECTED STATEMENT
    # ------------------------------------------------

    story.append(
        Paragraph("4. CORRECTED STATEMENT", heading_style)
    )

    correction = report.get("correction", {})

    corrected_claim = correction.get("corrected_claim")

    if corrected_claim:

        correction_table = Table(
            [[Paragraph(
                safe_text(corrected_claim),
                body_style
            )]],
            colWidths=[16 * cm]
        )

        correction_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1),
             colors.HexColor("#E2F0D9")),
            ("BOX", (0, 0), (-1, -1), 0.7, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 10)
        ]))

        story.append(correction_table)

    else:

        story.append(
            Paragraph(
                "No correction was generated.",
                body_style
            )
        )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "<b>Correction explanation:</b> "
            + safe_text(correction.get("explanation")),
            body_style
        )
    )

    story.append(
        Paragraph(
            "<b>Correction source:</b> "
            + safe_text(correction.get("correction_source")),
            body_style
        )
    )

    story.append(
        Paragraph(
            "<b>Source page:</b> "
            + safe_text(correction.get("source_page")),
            body_style
        )
    )

    # ------------------------------------------------
    # EVIDENCE SUMMARY
    # ------------------------------------------------

    story.append(
        Paragraph("5. EVIDENCE SUMMARY", heading_style)
    )

    evidence = report.get("evidence_summary", {})

    evidence_data = [
        ["Evidence Category", "Number of References"],
        [
            "Supporting",
            str(len(evidence.get("supporting_evidence", [])))
        ],
        [
            "Contradicting",
            str(len(evidence.get("contradicting_evidence", [])))
        ],
        [
            "Neutral",
            str(len(evidence.get("neutral_evidence", [])))
        ],
        [
            "Total Checked",
            str(evidence.get("total_references_checked", 0))
        ]
    ]

    evidence_table = Table(
        evidence_data,
        colWidths=[10 * cm, 6 * cm]
    )

    evidence_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))

    story.append(evidence_table)

    # ------------------------------------------------
    # DETAILED REFERENCE EVIDENCE
    # ------------------------------------------------

    story.append(
        Paragraph("6. DETAILED REFERENCE EVIDENCE", heading_style)
    )

    evidence_categories = [
        ("SUPPORTING EVIDENCE", "supporting_evidence"),
        ("CONTRADICTING EVIDENCE", "contradicting_evidence"),
        ("NEUTRAL EVIDENCE", "neutral_evidence")
    ]

    for category_title, category_key in evidence_categories:

        records = evidence.get(category_key, [])

        story.append(
            Paragraph(
                f"<b>{category_title}</b>",
                body_style
            )
        )

        if not records:

            story.append(
                Paragraph(
                    "No evidence available.",
                    small_style
                )
            )

        for item in records:

            evidence_text = safe_text(item.get("reference"))
            piece_id = safe_text(item.get("piece_id"))
            page = safe_text(item.get("page_number"))
            confidence = safe_text(
                item.get("nli_confidence", item.get("confidence"))
                )
            similarity = safe_text(item.get("similarity_score"))

            details = (
                f"<b>Piece ID:</b> {piece_id}<br/>"
                f"<b>Page:</b> {page}<br/>"
                f"<b>Similarity:</b> {similarity}<br/>"
                f"<b>NLI Classification:</b> "
                f"{safe_text(item.get('nli_classification', item.get('classification')))}<br/>"
                f"<b>Reference:</b> {evidence_text}"
            )

            story.append(
                Paragraph(details, small_style)
            )

            story.append(Spacer(1, 6))

    # ------------------------------------------------
    # ALL NLI VERIFICATION RESULTS
    # ------------------------------------------------

    story.append(
        Paragraph("7. ALL VERIFICATION RESULTS", heading_style)
    )

    all_results = report.get("all_verification_results", [])

    if all_results:

        results_data = [[
            "Rank",
            "Piece ID",
            "Page",
            "Similarity",
            "Classification",
            "Confidence"
        ]]

        for item in all_results:

            results_data.append([
                str(item.get("rank", "")),
                str(item.get("piece_id", "")),
                str(item.get("page_number", "")),
                str(item.get("similarity_score", "")),
                str(item.get("classification", "")),
                str(item.get("confidence", "")) + "%"
            ])

        results_table = Table(
            results_data,
            colWidths=[
                1.2 * cm,
                1.7 * cm,
                1.4 * cm,
                2.2 * cm,
                5 * cm,
                2.5 * cm
            ],
            repeatRows=1
        )

        results_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6)
        ]))

        story.append(results_table)

    else:

        story.append(
            Paragraph(
                "No verification results available.",
                body_style
            )
        )

    # ------------------------------------------------
    # DISCLAIMER
    # ------------------------------------------------

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "<b>Note:</b> This report presents automated model "
            "predictions based on the supplied reference document. "
            "NLI confidence scores are model outputs and should not "
            "be interpreted as guaranteed factual accuracy. "
            "The verdict depends on retrieval quality, model "
            "predictions, and configured thresholds.",
            small_style
        )
    )

    # ------------------------------------------------
    # PAGE FOOTER
    # ------------------------------------------------

    def add_page_number(canvas, doc):

        canvas.saveState()

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)

        canvas.drawString(
            2 * cm,
            1 * cm,
            "VeriFact AI | Evidence Report"
        )

        canvas.drawRightString(
            A4[0] - 2 * cm,
            1 * cm,
            f"Page {doc.page}"
        )

        canvas.restoreState()

    # ------------------------------------------------
    # GENERATE PDF
    # ------------------------------------------------

    print("\nGenerating PDF report...")

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    print("\n" + "=" * 60)
    print("VERIFACT AI - PDF REPORT GENERATED")
    print("=" * 60)

    print("\nPDF report created successfully!")
    print("Saved at:")
    print(output_path)

    print("\n" + "=" * 60)

    return output_path


# ------------------------------------------------
# FUNCTION FOR STREAMLIT INTEGRATION
# ------------------------------------------------

def generate_pdf_report(report, output_path):
    """
    Generate a PDF report directly from Streamlit report data.
    """

    return create_pdf_report(
        report=report,
        output_path=output_path
    )


# ------------------------------------------------
# STANDALONE EXECUTION
# ------------------------------------------------

if __name__ == "__main__":

    create_pdf_report()
