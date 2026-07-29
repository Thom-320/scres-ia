#!/usr/bin/env python3
"""Multi-page review PDF: figure audit + formula verification + deliverable status."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

D = Path(__file__).resolve().parent
OUT = D.parent / "REVIEW_figures_and_formulas_2026-07-23.pdf"
INK, MUTED, BLUE = "#1A1A1A", "#5A6570", "#2B6CB0"
plt.rcParams.update({"font.family": "Helvetica"})

FIGS = [
    ("fig1_mfsc_flow", "Figure M1 - MFSC 13-operation flow (thesis Fig. 6.2 modernized)",
     "AUDITED: all PT/Q/ROP values verified against thesis SS6.3.3; risk chips per op verified."),
    ("fig2_framework", "Figure M3 - Program Q experimental framework",
     "AUDITED: hyperparameters and 21-dim observation verified vs frozen contracts; overflow fixed."),
    ("fig3_ladder", "Comparator ladder (L0-L4)",
     "AUDITED: level semantics match the claim-boundary tables; geometry fixed."),
    ("fig4_results", "Program Q results by cell",
     "AUDITED: all 11 numbers verified PRESENT in Submission A source_of_truth.json."),
    ("fig5_ret_tree", "Figure M4 - ReT scoring branches",
     "AUDITED: 4 branches EXACT vs canonical excel_ret_value() (0/47,546 vs Excel); overlap fixed."),
    ("fig6_timeline", "Figure M2 - Episode structure",
     "AUDITED: offsets 24/72/120 h (batches), 30-150 h (orders), 1,344 h clearance verified."),
]

with PdfPages(OUT) as pdf:
    # ---- title page
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.90, "SCRES-IA - Figure & formula review", fontsize=22,
             weight="bold", color=INK)
    fig.text(0.08, 0.865, "C&IE manuscript deliverables - 23 July 2026", fontsize=12, color=BLUE)
    body = (
        "FORMULA VERIFICATION (the critical check)\n"
        "  All four ReT branches in the DOCX and in Figure M4 were verified against the\n"
        "  canonical implementation excel_ret_value() in supply_chain/garrido_replication.py\n"
        "  (the function that reproduces 47,546 workbook rows with zero mismatches):\n"
        "     risk active, AP>0            ->  Re_j = AP_j / LT            EXACT\n"
        "     risk active, AP=0, RP>0      ->  Re_j = 0.5 / RP_j           EXACT\n"
        "     risk active, no recovery     ->  Re_j = 0                    EXACT\n"
        "     no risk                      ->  Re_j = 1 - (B_tj+U_tj)/j    EXACT\n\n"
        "NUMBER VERIFICATION\n"
        "  The 11 result numbers in Figure 4 (deltas, LCBs, latencies) are all present in\n"
        "  papers/submission_a_program_q/source_of_truth.json (Submission A branch).\n\n"
        "ELSEVIER COMPLIANCE APPLIED\n"
        "  Double-column width 185 mm; minimum lettering >= 6 pt; line work >= 0.25 pt;\n"
        "  vector PDF (fonts embedded, Type 42) + 300 dpi PNG; palette passes the six\n"
        "  colorblind-safety checks (validated by script, not by eye).\n\n"
        "DELIVERABLES\n"
        "  v0_neuralNet-scres_DES_section_updated.docx  (v.0 with Section 3.3 replaced;\n"
        "    style preserved; 4 figures embedded; KAN section and all other content intact)\n"
        "  deliverables/figures/*.pdf|png  (6 publication figures + this review)\n\n"
        "KNOWN LIMIT\n"
        "  Figure M1 omits the R3 black-swan chips shown in thesis Fig. 6.2 on Op5/6/7/9\n"
        "  (space); add before submission if the risk-active experiments are included."
    )
    fig.text(0.08, 0.83, body, fontsize=9.3, color=INK, va="top", family="Courier New")
    pdf.savefig(fig); plt.close(fig)

    # ---- side-by-side fidelity page: thesis Fig 6.2 vs our Figure M1
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
    axes[0].imshow(mpimg.imread(D / "thesis_fig62_reference.png")); axes[0].axis("off")
    axes[0].set_title("Thesis Figure 6.2 (Garrido-Rios 2017, p. 88) - source reference",
                      fontsize=10, color=MUTED)
    axes[1].imshow(mpimg.imread(D / "fig1_mfsc_flow.png")); axes[1].axis("off")
    axes[1].set_title("Our Figure M1 - same topology, PT/Q/ROP and per-op risks preserved",
                      fontsize=10, color=BLUE)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # ---- one page per figure
    for name, title, audit in FIGS:
        img = mpimg.imread(D / f"{name}.png")
        h, w = img.shape[:2]
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes([0.03, 0.10, 0.94, 0.80])
        ax.imshow(img); ax.axis("off")
        fig.text(0.03, 0.945, title, fontsize=13, weight="bold", color=INK)
        fig.text(0.03, 0.045, audit, fontsize=9, color=MUTED, style="italic")
        pdf.savefig(fig); plt.close(fig)

print("review pdf:", OUT)
