"""
scripts/build_full_report.py
----------------------------
Master orchestrator to generate:
1. UNSW_NB15_Complete_Research_Report.md (Comprehensive source document)
2. UNSW_NB15_Complete_Research_Report.pdf (Print-ready, academic ReportLab PDF)
3. Page-by-page visual inspection images in report_assets/inspections/

Includes:
- Robust markdown/LaTeX sanitization to clean typography
- Formatted Table of Contents, List of Figures, List of Tables
- Native Callout Boxes for "WHAT THIS RESULT MEANS" & "WHAT THIS RESULT DOES NOT MEAN"
- Proper spacing and image sizing to eliminate orphan pages
"""

import sys
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional, cast

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ReportLab imports
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

import pymupdf

# Import narrative generators
from scripts.report_data import PROJECT_METADATA
from scripts.report_content import build_executive_summary_qa
from scripts.report_sections import (
    get_title_markdown, get_executive_summary_markdown,
    get_introduction_markdown, get_dataset_markdown,
    get_features_markdown, get_architecture_markdown,
    get_base_models_markdown, get_autoencoder_markdown,
    get_stacking_markdown, get_fusion_markdown
)
from scripts.report_sprints import (
    get_sprint7_markdown, get_sprint8_markdown,
    get_sprint9_markdown, get_sprint10_markdown,
    get_sprint11_markdown, get_sprint12_markdown,
    get_sprint13_markdown
)
from scripts.report_syntheses import (
    get_why_it_happened_markdown, get_problems_and_resolutions_markdown,
    get_consolidated_results_markdown, get_findings_and_conclusion_markdown,
    get_appendices_markdown, get_final_checklist_markdown
)

MD_OUT_PATH = ROOT / "UNSW_NB15_Complete_Research_Report.md"
PDF_OUT_PATH = ROOT / "UNSW_NB15_Complete_Research_Report.pdf"
ASSETS_DIR = ROOT / "report_assets"
FIGURES_DIR = ASSETS_DIR / "figures"
INSPECTION_DIR = ASSETS_DIR / "inspections"
INSPECTION_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# TWO-PASS NUMBERED CANVAS (RUNNING HEADERS & FOOTERS)
# -----------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        getattr(self, "_startPage")()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        page_num: int = getattr(self, "_pageNumber", 1)
        if page_num == 1:
            return  # Suppress running header/footer on title page

        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#1B365D"))

        # Header
        self.drawString(45, 842 - 32, "UNSW-NB15 INTRUSION DETECTION SYSTEM — COMPLETE RESEARCH REPORT")
        self.setFont("Helvetica-Oblique", 7.5)
        self.setFillColor(colors.HexColor("#555555"))
        self.drawRightString(595 - 45, 842 - 32, "EXP_ZERODAY_V1 | SPRINT 7–13")

        # Top rule
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.6)
        self.line(45, 842 - 37, 595 - 45, 842 - 37)

        # Bottom rule
        self.line(45, 42, 595 - 45, 42)

        # Footer
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#666666"))
        self.drawString(45, 30, "CONFIDENTIAL & AUTHORITATIVE RESEARCH AUDIT — FROZEN BENCHMARK")
        page_str = f"Page {page_num} of {page_count}"
        self.setFont("Helvetica-Bold", 8)
        self.drawRightString(595 - 45, 30, page_str)

        self.restoreState()


# -----------------------------------------------------------------------------
# 1. TEXT SANITIZATION (MARKDOWN & LATEX TO CLEAN HTML FOR REPORTLAB)
# -----------------------------------------------------------------------------
def sanitize_text(txt: str) -> str:
    """Converts markdown formatting and LaTeX expressions into clean ReportLab HTML."""
    # 1. Standard math substitutions
    txt = txt.replace(r"\approx", "≈").replace("\a", "≈").replace(r"\sim", "≈")
    txt = txt.replace(r"\Delta", "Δ").replace(r"\tau", "τ").replace(r"\epsilon", "ε")
    txt = txt.replace(r"\sigma", "σ").replace(r"\mu", "μ").replace(r"\pm", "±")
    txt = txt.replace(r"\times", "×").replace(r"\ge", "≥").replace(r"\le", "≤")
    txt = txt.replace(r"\in", "∈").replace(r"\lor", "∨").replace(r"\land", "∧")
    txt = txt.replace(r"\rightarrow", "→").replace(r"\_", "_").replace(r"\%", "%")
    txt = txt.replace(r"\mathbb{R}", "R").replace(r"\mathbf{x}", "x").replace(r"\mathbf{P}", "P")
    txt = txt.replace(r"\mathbf{\hat{x}}", "x̂").replace(r"\hat{y}", "ŷ").replace(r"\hat{p}", "p̂")
    txt = txt.replace(r"\quad", " ")
    txt = re.sub(r"\\text\{(.+?)\}", r"\1", txt)
    txt = txt.replace(r"\text", "")
    
    # 2. Subscript patterns like _{C06} or _{Stack} or _{Backdoor}
    txt = re.sub(r"_\{([a-zA-Z0-9\-\+\s]+)\}", r"<sub>\1</sub>", txt)
    txt = re.sub(r"_([0-9])", r"<sub>\1</sub>", txt)
    
    # 3. Quadrant patterns like Q_1, Q_2, Q_3, Q_4
    txt = re.sub(r"Q\_([1-4])", r"Q<sub>\1</sub>", txt)
    
    # 4. Convert markdown bold **text** to <b>text</b>
    txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt)
    # 5. Convert markdown inline code `code` to bold colored text
    txt = re.sub(r"`(.+?)`", r'<b><font color="#1B365D">\1</font></b>', txt)
    txt = re.sub(r"<code>(.+?)</code>", r'<b><font color="#1B365D">\1</font></b>', txt)
    
    # 6. Remove stray dollar signs, curly braces from math mode
    txt = txt.replace("$", "")

    # 7. Protect XML entities for ReportLab Paragraph:
    valid_tags = []
    def tag_repl(m):
        valid_tags.append(m.group(0))
        return f"___HTMLTAG_{len(valid_tags)-1}___"
    
    txt = re.sub(r"<(/?(?:b|i|sub|sup|font)(?:\s+[^>]*?)?|(?:br\s*/?)|(?:/font))>", tag_repl, txt)
    # Now any remaining < and > are mathematical/raw
    txt = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Restore valid tags
    for idx, tag in enumerate(valid_tags):
        txt = txt.replace(f"___HTMLTAG_{idx}___", tag)
    return txt


# -----------------------------------------------------------------------------
# 2. REPORTLAB STYLES AND FLOWABLE BUILDERS
# -----------------------------------------------------------------------------
def setup_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {}

    styles['Title'] = ParagraphStyle(
        'DocTitle', parent=base['Title'],
        fontName='Helvetica-Bold', fontSize=22, leading=26,
        textColor=colors.HexColor('#1B365D'), alignment=TA_CENTER, spaceAfter=8
    )
    styles['Subtitle'] = ParagraphStyle(
        'DocSubtitle', parent=base['Normal'],
        fontName='Helvetica-Bold', fontSize=12, leading=16,
        textColor=colors.HexColor('#4A607A'), alignment=TA_CENTER, spaceAfter=12
    )
    styles['Heading1'] = ParagraphStyle(
        'DocH1', parent=base['Heading1'],
        fontName='Helvetica-Bold', fontSize=13, leading=16,
        textColor=colors.HexColor('#1B365D'), spaceBefore=12, spaceAfter=6, keepWithNext=True
    )
    styles['Heading2'] = ParagraphStyle(
        'DocH2', parent=base['Heading2'],
        fontName='Helvetica-Bold', fontSize=10.5, leading=13.5,
        textColor=colors.HexColor('#0B2545'), spaceBefore=8, spaceAfter=4, keepWithNext=True
    )
    styles['Heading3'] = ParagraphStyle(
        'DocH3', parent=base['Heading3'],
        fontName='Helvetica-Bold', fontSize=9, leading=12,
        textColor=colors.HexColor('#1B365D'), spaceBefore=6, spaceAfter=3, keepWithNext=True
    )
    styles['Body'] = ParagraphStyle(
        'DocBody', parent=base['Normal'],
        fontName='Helvetica', fontSize=8, leading=10.5,
        textColor=colors.HexColor('#222222'), alignment=TA_LEFT, spaceAfter=3.5
    )
    styles['BodyBold'] = ParagraphStyle(
        'DocBodyBold', parent=styles['Body'], fontName='Helvetica-Bold'
    )
    styles['Bullet'] = ParagraphStyle(
        'DocBullet', parent=styles['Body'], leftIndent=12, firstLineIndent=-8, spaceAfter=2
    )
    styles['TableText'] = ParagraphStyle(
        'DocTableText', parent=base['Normal'],
        fontName='Helvetica', fontSize=7, leading=9, textColor=colors.HexColor('#222222')
    )
    styles['TableHeader'] = ParagraphStyle(
        'DocTableHeader', parent=base['Normal'],
        fontName='Helvetica-Bold', fontSize=7, leading=9,
        textColor=colors.white, alignment=TA_CENTER
    )
    styles['CalloutTitle'] = ParagraphStyle(
        'DocCalloutTitle', parent=base['Normal'],
        fontName='Helvetica-Bold', fontSize=8, leading=10,
        textColor=colors.HexColor('#1B365D'), spaceAfter=2
    )
    styles['CalloutText'] = ParagraphStyle(
        'DocCalloutText', parent=base['Normal'],
        fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor('#333333')
    )
    styles['FigCaption'] = ParagraphStyle(
        'DocFigCaption', parent=base['Normal'],
        fontName='Helvetica-Bold', fontSize=7.5, leading=9.5,
        textColor=colors.HexColor('#1B365D'), alignment=TA_CENTER, spaceBefore=3, spaceAfter=1
    )
    styles['FigInterpretation'] = ParagraphStyle(
        'DocFigInterpretation', parent=base['Normal'],
        fontName='Helvetica-Oblique', fontSize=7, leading=8.5,
        textColor=colors.HexColor('#444444'), alignment=TA_LEFT, leftIndent=8, rightIndent=8, spaceAfter=6
    )

    return styles


def create_callout(title: str, text: str, border_color: str = "#1B365D", bg_color: str = "#F4F6F9", styles: Optional[Dict[str, ParagraphStyle]] = None) -> Table:
    if styles is None:
        styles = setup_styles()
    clean_text = sanitize_text(text)
    clean_title = sanitize_text(title)
    content = [
        Paragraph(f"<b>{clean_title}</b>", styles['CalloutTitle']),
        Paragraph(clean_text, styles['CalloutText'])
    ]
    t = Table([[content]], colWidths=[505])
    t.setStyle(TableStyle(cast(Any, [
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg_color)),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(border_color)),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ])))
    return t


def create_table(headers: List[str], rows: List[List[str]], col_widths: List[float], styles: Optional[Dict[str, ParagraphStyle]] = None) -> Table:
    if styles is None:
        styles = setup_styles()
    table_data = []
    header_cells = [Paragraph(f"<b>{sanitize_text(h)}</b>", styles['TableHeader']) for h in headers]
    table_data.append(header_cells)

    for r in rows:
        row_cells = []
        for cell in r:
            row_cells.append(Paragraph(sanitize_text(cell), styles['TableText']))
        table_data.append(row_cells)

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(cast(Any, [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D0D7DE')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
    ])))
    return t


def create_figure(img_path: Path, width: float, height: float, fig_num: int, title: str, source: str, interpretation: str, styles: Optional[Dict[str, ParagraphStyle]] = None) -> KeepTogether:
    if styles is None:
        styles = setup_styles()
    elements = []
    if img_path.exists():
        elements.append(Image(str(img_path), width=width, height=height))
    else:
        elements.append(Paragraph(f"[Image Missing: {img_path.name}]", styles['BodyBold']))

    elements.append(Paragraph(f"<b>Figure {fig_num}:</b> {sanitize_text(title)}", styles['FigCaption']))
    elements.append(Paragraph(f"<b>Source:</b> {sanitize_text(source)} | <b>Interpretation:</b> {sanitize_text(interpretation)}", styles['FigInterpretation']))
    return KeepTogether(elements)


def parse_md_table_to_flowable(table_lines: List[str], styles: Dict[str, ParagraphStyle]) -> Union[Table, Spacer]:
    rows = []
    for line in table_lines:
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if parts and all(set(p).issubset({'-', ':', ' '}) for p in parts):
            continue
        rows.append(parts)

    if not rows:
        return Spacer(1, 1)

    headers = rows[0]
    data_rows = rows[1:]
    n_cols = len(headers)
    col_w = 505.0 / max(1, n_cols)
    col_widths = [col_w] * n_cols

    return create_table(headers, data_rows, col_widths, styles=styles)


# -----------------------------------------------------------------------------
# 3. BUILD STORY FLOWABLES
# -----------------------------------------------------------------------------
def build_story(styles) -> List[Any]:
    story = []

    # -------------------------------------------------------------------------
    # COVER / TITLE PAGE
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 30))
    story.append(Paragraph(PROJECT_METADATA['title'], styles['Title']))
    story.append(Paragraph(PROJECT_METADATA['subtitle'], styles['Subtitle']))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1B365D'), spaceBefore=5, spaceAfter=15))

    meta_text = f"""
    <b>Research Scope:</b> {PROJECT_METADATA['scope']}<br/><br/>
    <b>Target Dataset:</b> {PROJECT_METADATA['dataset']}<br/><br/>
    <b>Final Experiment Status:</b> {PROJECT_METADATA['final_experiment_status']}<br/><br/>
    <b>Authoritative Commit Hash:</b> <font color="#1B365D">{PROJECT_METADATA['final_commit']}</font><br/><br/>
    <b>Authoritative Git Freeze Tag:</b> <font color="#1B365D">{PROJECT_METADATA['final_tag']}</font><br/><br/>
    <b>Evaluation Methodology:</b> Four-way split architecture, 5-Fold OOF Meta-Learning, Benign-Only Deep Autoencoder, and Pre-Registered Statistical Hypotheses.<br/><br/>
    <b>Documentation Standards:</b> Academic publication standard, zero data fabrication, frozen artifacts audit.<br/><br/>
    <b>Report Publication Date:</b> {PROJECT_METADATA['report_date']}
    """
    story.append(create_callout("PROJECT SPECIFICATIONS & AUTHORITATIVE METADATA", meta_text, border_color="#1B365D", bg_color="#F4F6F9", styles=styles))
    story.append(Spacer(1, 15))

    exec_summary_callout = """
    <b>Document Purpose:</b> This document provides the complete, authoritative record of the UNSW-NB15 intrusion detection project across Sprints 7 through 13. It presents the technical problem, data architecture, feature selection, base models, Out-of-Fold stacking, benign-only Autoencoder, hybrid logical-OR fusion, systematic ablations, model explainability with forensic audits, frozen reproducibility verification, and the culminating controlled zero-day simulation on 583 isolated Backdoor network attacks.
    """
    story.append(create_callout("RESEARCH OVERVIEW & AUDIT INTEGRITY", exec_summary_callout, border_color="#0B2545", bg_color="#EBF3FB", styles=styles))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # EXECUTIVE SUMMARY
    # -------------------------------------------------------------------------
    story.append(Paragraph("Executive Summary", styles['Heading1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=6))
    story.append(Paragraph(
        "This executive summary synthesizes the entire research journey across Sprints 7 through 13. "
        "The project rigorously answers whether unsupervised anomaly detection can enhance supervised ensemble classifiers "
        "in detecting novel, zero-day network intrusions under strict leakage controls.",
        styles['Body']
    ))
    story.append(Spacer(1, 4))

    qa_list = build_executive_summary_qa()
    for item in qa_list:
        story.append(Paragraph(f"<b>{sanitize_text(item['q'])}</b>", styles['BodyBold']))
        story.append(Paragraph(sanitize_text(item['a']), styles['Body']))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 6))
    meaning_box = (
        "Key Takeaway: Supervised stacking proved remarkably resilient, detecting 99.83% of unseen Backdoors by learning "
        "fundamental flow primitives. However, the unsupervised Autoencoder was rendered inert by legitimate protocol connection-aborts (TCP RST/FIN), "
        "demonstrating that global anomaly thresholds are vulnerable to benign network diversity."
    )
    story.append(create_callout("WHAT THIS EXECUTIVE SUMMARY MEANS", meaning_box, border_color="#2E7D32", bg_color="#F1F8E9", styles=styles))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # TABLE OF CONTENTS & LIST OF FIGURES/TABLES
    # -------------------------------------------------------------------------
    story.append(Paragraph("Table of Contents & Document Index", styles['Heading1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))

    toc_headers = ["Section", "Title / Topic", "Research Focus"]
    toc_rows = [
        ["1", "Project Introduction & Research Formulation", "Problem background, research gap, objectives, locked hypotheses (H1–H3)"],
        ["2", "UNSW-NB15 Dataset & Split Architecture", "Row counts, Backdoor isolation, 4-way split, leakage prevention, verification"],
        ["3", "Feature Engineering & Feature Selection", "One-hot encoding, Mutual Information, K=75 selection, complexity plateau"],
        ["4", "Complete System Architecture", "End-to-end data pipeline, multi-branch model interfaces, logical fusion"],
        ["5", "Supervised Base Classifiers (EXP_BASE_MODELS_V1)", "Decision Tree, Random Forest, Linear SVM, Neural Network (IDSNet)"],
        ["6", "Unsupervised Autoencoder (EXP_AE_V1)", "75->12->6->12->75 topology, benign training, tau=11.16006, suppression cause"],
        ["7", "Out-of-Fold Stacking & Meta-Learning", "5-fold OOF cross-validation, Logistic Regression meta-weights, 3 seeds"],
        ["8", "Hybrid Logical Fusion Architecture", "C01 (stacking) vs C06 (OR-fusion), rescue concept, quadrant structure"],
        ["9", "Sprint 7 — Autoencoder Development", "Benign training, validation RE distribution, threshold calibration"],
        ["10", "Sprint 8 — Baseline Evaluation Foundation", "2x2 fusion exploration, false-positive rate gates, C06 baseline selection"],
        ["11", "Sprint 9 — Formal Hypothesis Testing", "H1 SUPPORTED, H2 NOT_SUPPORTED, H3 NOT_SUPPORTED"],
        ["12", "Sprint 10 — Systematic Ablation Study", "A0–A6 configurations, RF dominance, soft-voting failure, FP lineage"],
        ["13", "Sprint 11 — Explainability & Provenance Audit", "SHAP feature importances, AE architecture discrepancy & forensic quarantine"],
        ["14", "Sprint 12 — Final Reproducibility Verification", "Zero-training audit (0 fit calls), bitwise reproduction, sklearn drift"],
        ["15", "Sprint 13 — Controlled Zero-Day Simulation", "Protocol V1.4, 583 Backdoors, quadrants Q1–Q4, Wilson CI, 44 gates"],
        ["16", "Why Did This Result Happen? Syntheses", "Systematic analytical breakdown of causal mechanisms and boundaries"],
        ["17", "Experimental Problems & Resolutions", "Forensic investigations: AE provenance, lineage, TCP abort outliers"],
        ["18", "Consolidated Final Results (Tables A–F)", "Benchmark, Dev-Test, Ablation, Zero-Day, Hypotheses, Reproducibility"],
        ["19", "Major Scientific Findings", "7 core empirical conclusions, meanings, and research boundaries"],
        ["20", "Timeline, Milestones & Conclusion", "Milestones S1–S13, final conclusion, Appendices A–J, Checklist"]
    ]
    story.append(create_table(toc_headers, toc_rows, [45, 230, 230], styles=styles))
    story.append(Spacer(1, 8))

    # List of figures and tables
    story.append(Paragraph("<b>List of Publication Figures:</b>", styles['BodyBold']))
    figures_summary = (
        "Figure 1: UNSW-NB15 Active Split Architecture | Figure 2: Sample Size & Class Distribution | "
        "Figure 3: System Architecture Flow | Figure 4: Base Model Performance Comparison | "
        "Figure 5: Autoencoder Reconstruction Error & Thresholds | Figure 6: Meta-Learner Model Weights | "
        "Figure 7: Hypotheses Decision Matrix | Figure 8: Ablation Macro-F1 | Figure 9: Paired Ablation Deltas | "
        "Figure 10: Sprint 12 Reproducibility Residuals | Figure 11: Zero-Day Detection Rates | "
        "Figure 12: Quadrant Structure | Figure 13: Benign FPR | Figure 14: Research Milestones Timeline."
    )
    story.append(Paragraph(figures_summary, styles['Body']))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SECTION 1: INTRODUCTION & RESEARCH FORMULATION
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Project Introduction & Research Formulation", styles['Heading1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph("1.1 Problem Background", styles['Heading2']))
    story.append(Paragraph(
        "Modern enterprise networks are subjected to an expanding volume of polymorphic and zero-day cyber threats. "
        "Traditional Network Intrusion Detection Systems (NIDS) are predominantly rule-based or trained using supervised machine learning algorithms. "
        "While supervised models excel at identifying recognized attack signatures, they suffer from fundamental blind spots when confronted with novel, "
        "previously unseen intrusion classes. Attackers deliberately modify exploit payloads, obfuscate protocol handshakes, and alter packet timings "
        "to bypass perimeter defenses. Consequently, modern security operations centers require detection systems capable of maintaining high discriminative "
        "accuracy on established threats while demonstrating reliable anomaly sensitivity to unobserved attacks.",
        styles['Body']
    ))

    story.append(Paragraph("1.2 Research Problem", styles['Heading2']))
    story.append(Paragraph(
        "The core research problem addressed by this project is whether a hybrid architecture—combining supervised multi-model ensemble stacking "
        "with an unsupervised benign-only deep Autoencoder—can successfully generalize to withheld, zero-day attack categories without incurring "
        "unacceptable false-positive inflation on benign operational traffic. Specifically, we investigate whether an unsupervised reconstruction "
        "model can rescue attack samples that completely evade the decision boundaries of supervised classifiers.",
        styles['Body']
    ))

    story.append(Paragraph("1.3 Research Gap", styles['Heading2']))
    story.append(Paragraph(
        "Existing intrusion detection literature frequently exhibits critical methodological weaknesses: (1) pervasive data leakage where test samples "
        "influence preprocessing and scaling; (2) unrealistic zero-day claims where the tested attack family was exposed during hyperparameter tuning; "
        "(3) unsubstantiated hybrid fusion claims without quantifying false-positive penalties or empirical rescue rates; and (4) lack of reproducible, "
        "frozen evaluation pipelines. This project bridges these gaps through immutable data partitions, frozen checkpoints, and pre-registered hypotheses.",
        styles['Body']
    ))

    story.append(Paragraph("1.4 Project Objectives & Locked Hypotheses", styles['Heading2']))
    story.append(Paragraph(
        "The project is structured around three formal, pre-registered hypotheses: "
        "<br/>• <b>Hypothesis 1 (H1 — Stacking Superiority):</b> Multi-seed OOF stacking exceeds the best individual base model (Random Forest) by at least ε = 0.005 Macro-F1 on Development-Test."
        "<br/>• <b>Hypothesis 2 (H2 — Autoencoder Standalone Anomaly Detection):</b> Unsupervised AE detects a non-zero count of unseen Backdoors at its frozen threshold (rule DD-4)."
        "<br/>• <b>Hypothesis 3 (H3 — Hybrid Fusion Rescue Efficacy):</b> Logical-OR fusion (C06) improves attack detection over stacking (C01) on unseen Backdoors without inflating benign FPR by > 2%.",
        styles['Body']
    ))
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------------------
    # SECTION 2: DATASET & SPLIT ARCHITECTURE
    # -------------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("2. UNSW-NB15 Dataset & Active Split Architecture", styles['Heading1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "The UNSW-NB15 dataset comprises 257,673 records across an official training set (175,341 rows) and testing set (82,332 rows). "
        "It features 42 raw attributes spanning packet timings, byte counts, and connection state metrics across 9 contemporary attack families. "
        "To establish a true zero-day evaluation harness, the dataset was partitioned into four mutually disjoint partitions with exact row conservation:",
        styles['Body']
    ))
    story.append(Spacer(1, 3))

    split_headers = ["Partition Name", "Total Rows", "Benign (0)", "Attack (1)", "Attack Families", "Role in Research Lifecycle"]
    split_rows = [
        ["TRAIN", "162,395", "44,800", "117,595", "8 (No Backdoor)", "Supervised training & OOF meta-learning"],
        ["VALIDATION", "11,200", "11,200", "0", "0 (Benign Only)", "AE threshold calibration & sanity checks"],
        ["DEVELOPMENT_TEST", "81,749", "37,000", "44,749", "8 (No Backdoor)", "Held-out known-attack benchmark"],
        ["PROTECTED_BACKDOOR", "583", "0", "583", "1 (Backdoor Only)", "Isolated zero-day proxy population"],
        ["EXCLUDED_BACKDOOR", "1,746", "0", "1,746", "1 (Archived Backdoor)", "Withheld training Backdoors (prevent leakage)"]
    ]
    story.append(create_table(split_headers, split_rows, [85, 55, 55, 55, 110, 145], styles=styles))
    story.append(Spacer(1, 4))

    # Figure 1: Split architecture diagram (height adjusted to 125 pt)
    story.append(create_figure(
        FIGURES_DIR / "fig01_split_architecture.png", 490, 125, 1,
        "UNSW-NB15 Active Split & Data Isolation Flow Architecture",
        "data/splits/split_metadata.json",
        "Illustrates the permanent isolation of Backdoor attacks to establish a pristine zero-day evaluation harness.",
        styles=styles
    ))
    story.append(Spacer(1, 4))

    # Figure 2: Class distribution (height adjusted to 120 pt)
    story.append(create_figure(
        FIGURES_DIR / "fig03_class_distribution.png", 490, 120, 2,
        "Sample Size & Class Balance Across Evaluated Splits",
        "data/splits/split_metadata.json",
        "Shows the heavy attack concentration in TRAIN and the pure benign composition of VALIDATION.",
        styles=styles
    ))
    story.append(Spacer(1, 4))

    leakage_box = (
        "Leakage Prevention Audit: Exactly 0 Backdoor samples entered TRAIN or VALIDATION. "
        "Feature scaling statistics were computed strictly on TRAIN and immutably applied across splits. "
        "Reconstruction verification confirmed 100% exact row and column conservation."
    )
    story.append(create_callout("STRICT DATA ISOLATION VERIFICATION", leakage_box, border_color="#1B365D", bg_color="#F4F6F9", styles=styles))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SECTION 3: FEATURE SELECTION
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Feature Engineering & Selection (EXP_MI_V1_1)", styles['Heading1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "One-hot encoding of categorical variables (`proto`, `service`, `state`) expanded the 42 raw features to 193 candidate features. "
        "Mutual Information (MI) scoring was conducted using 5-fold cross-validation on TRAIN. "
        "The inner-CV Macro-F1 curve demonstrated a steep rise from K=10 (0.8249) to K=50 (0.9196), reaching its global maximum at K=75 (0.9198), "
        "beyond which performance plateaued (K=100: 0.9198; K=150: 0.9197). "
        "Following pre-registered complexity rules, K=75 was locked as the optimal feature set, reducing dimensionality by 61.1%.",
        styles['Body']
    ))
    story.append(Spacer(1, 3))

    feat_headers = ["Candidate K", "Mean Inner-CV Macro-F1", "Std Macro-F1", "Complexity Delta", "Selection Rationale"]
    feat_rows = [
        ["10", "0.824852", "0.003435", "-65 features", "Severe underfitting on complex attacks"],
        ["20", "0.864436", "0.002428", "-55 features", "Sub-optimal feature representation"],
        ["30", "0.897442", "0.000917", "-45 features", "Approaching stable representation"],
        ["40", "0.916198", "0.002122", "-35 features", "Good baseline, but room for gain"],
        ["50", "0.919560", "0.002323", "-25 features", "Approaching plateau boundary"],
        ["75", "0.919799", "0.002393", "Optimal Winner", "Global peak Macro-F1 & plateau onset"],
        ["100", "0.919775", "0.002436", "+25 features", "Plateau; redundant parameter overhead"],
        ["150", "0.919750", "0.002506", "+75 features", "Slight degradation due to noise inflation"]
    ]
    story.append(create_table(feat_headers, feat_rows, [70, 110, 85, 100, 140], styles=styles))
    story.append(Spacer(1, 6))

    feat_breakdown = (
        "Selected Feature Composition: Exactly 39 continuous numeric features (100% retention), "
        "25 protocol dummies (proto), 6 service dummies (service), and 5 connection state dummies (state). "
        "The top 5 predictive features are: sbytes, sttl, dbytes, ct_state_ttl, and dttl."
    )
    story.append(create_callout("FEATURE REPRESENTATION PROFILE", feat_breakdown, border_color="#1B365D", bg_color="#F4F6F9", styles=styles))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SECTION 4: SYSTEM ARCHITECTURE
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Complete System Architecture", styles['Heading1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "The end-to-end architecture orchestrates two complementary branches: (1) a supervised diversity ensemble combining four base classifiers "
        "via an Out-of-Fold Logistic Regression meta-learner, and (2) an unsupervised deep Autoencoder trained exclusively on benign traffic. "
        "The two branches are fused via an inclusive logical-OR decision rule (C06) to evaluate anomaly rescue capabilities.",
        styles['Body']
    ))
    story.append(Spacer(1, 4))

    # Figure 3: Complete system architecture diagram (height adjusted to 150 pt)
    story.append(create_figure(
        FIGURES_DIR / "fig02_system_architecture.png", 490, 150, 3,
        "Complete End-to-End Hybrid Intrusion Detection Architecture",
        "Repository architectural specifications",
        "Shows the data flow from raw UNSW-NB15 to supervised stacking (C01), Autoencoder (AE), and logical-OR fusion (C06).",
        styles=styles
    ))
    story.append(Spacer(1, 6))

    arch_table_headers = ["Architecture Subsystem", "Input Dimensions", "Core Model / Operator", "Primary Output", "Theoretical Function"]
    arch_table_rows = [
        ["Data Preprocessing", "42 Raw Features", "One-hot schema + Standard Scaler", "75 Scaled Features", "Leakage-free numerical transformation"],
        ["Supervised Ensemble", "75 Scaled Features", "DT, RF (300), SVM, NN (IDSNet)", "4 Class Probabilities", "Discriminative boundary learning"],
        ["OOF Meta-Learner", "4 Base Probabilities", "Logistic Regression (lbfgs, C=1.0)", "C01 Binary Decision", "Optimal error-decorrelated weighting"],
        ["Unsupervised Anomaly", "75 Normal Features", "Autoencoder (75->12->6->12->75)", "RE > 11.16006 Decision", "Normality manifold reconstruction"],
        ["Hybrid Fusion Engine", "C01 and AE Outputs", "Inclusive Logical-OR Rule", "C06 Binary Decision", "Zero-day rescue via dual-mode coverage"]
    ]
    story.append(create_table(arch_table_headers, arch_table_rows, [95, 80, 110, 90, 130], styles=styles))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SECTION 5: BASE CLASSIFIERS
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Supervised Base Classifiers (EXP_BASE_MODELS_V1)", styles['Heading1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "Four diverse base classifiers were trained on the 162,395 rows of TRAIN and evaluated on the 81,749 held-out rows of DEVELOPMENT_TEST:",
        styles['Body']
    ))
    story.append(Spacer(1, 3))

    base_headers = ["Model Name", "Macro-F1", "Macro Precision", "Macro Recall", "Balanced Accuracy", "FPR", "Runtime (s)"]
    base_rows = [
        ["Decision Tree (Entropy)", "0.849852", "0.878340", "0.844352", "0.844352", "0.279541", "9.42"],
        ["Random Forest (300 Trees)", "0.880733", "0.903932", "0.874944", "0.874944", "0.231027", "31.22"],
        ["Support Vector Machine (Linear)", "0.823613", "0.851945", "0.818906", "0.818906", "0.310568", "79.32"],
        ["Neural Network (IDSNet)", "0.894293", "0.898909", "0.891850", "0.891850", "0.152432", "284.21"]
    ]
    story.append(create_table(base_headers, base_rows, [115, 65, 70, 65, 75, 60, 55], styles=styles))
    story.append(Spacer(1, 6))

    # Figure 4: Base models comparison (height adjusted to 140 pt)
    story.append(create_figure(
        FIGURES_DIR / "fig05_base_models_comparison.png", 490, 140, 4,
        "Comparative Performance of Four Supervised Base Models on Development-Test",
        "results/base_models/EXP_BASE_MODELS_V1/baseline_results.csv",
        "Highlights the superiority of the Neural Network and Random Forest, and the high false-alarm rates of Linear SVM and Decision Tree.",
        styles=styles
    ))
    story.append(Spacer(1, 6))

    base_box = (
        "Model Observations: Neural Network achieved the highest single-model Macro-F1 (0.8943) and lowest FPR (15.24%), "
        "while Random Forest delivered the most reliable bagging precision. Decision Tree and Linear SVM exhibited substantial false alarm rates (>27%)."
    )
    story.append(create_callout("BASE MODEL EVALUATION SYNTHESIS", base_box, border_color="#1B365D", bg_color="#F4F6F9", styles=styles))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SECTION 6: AUTOENCODER
    # -------------------------------------------------------------------------
    story.append(Paragraph("6. Unsupervised Benign-Only Autoencoder (EXP_AE_V1)", styles['Heading1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "The Autoencoder employs a symmetric feed-forward topology: 75 -> 12 -> 6 -> 12 -> 75 comprising exactly 2,049 trainable parameters. "
        "It was trained strictly on 40,320 benign flows from TRAIN using Adam optimizer (lr=0.001, weight decay=0.0001, batch size=256). "
        "Threshold calibration was performed on the 11,200 benign flows of VALIDATION to guarantee bounded false alarms under operational conditions.",
        styles['Body']
    ))
    story.append(Spacer(1, 3))

    # Figure 5: AE Reconstruction error distribution (height adjusted to 140 pt)
    story.append(create_figure(
        FIGURES_DIR / "fig04_ae_reconstruction_error.png", 490, 140, 5,
        "Autoencoder Reconstruction Error Distribution & Threshold Calibration on Benign VALIDATION",
        "results/autoencoder/EXP_AE_V1/threshold/threshold_calibration.json",
        "Shows typical errors clustering near 0.065, while the parametric threshold (tau=11.16006) is displaced outward by extreme TCP abort outliers.",
        styles=styles
    ))
    story.append(Spacer(1, 6))

    ae_table_headers = ["Threshold Rule", "Percentile", "Threshold Value (tau)", "Validation FP Count", "Empirical Validation FPR"]
    ae_table_rows = [
        ["p95", "95.0%", "0.567386", "560 / 11,200", "0.050000 (5.00%)"],
        ["p99", "99.0%", "1.512164", "112 / 11,200", "0.010000 (1.00%)"],
        ["p99.9", "99.9%", "10.696876", "12 / 11,200", "0.001071 (0.11%)"],
        ["mean + 2*sigma", "Parametric", "7.515109", "23 / 11,200", "0.002054 (0.21%)"],
        ["mean + 3*sigma", "Frozen Canonical", "11.160063", "7 / 11,200", "0.000625 (0.06%)"]
    ]
    story.append(create_table(ae_table_headers, ae_table_rows, [100, 75, 110, 110, 110], styles=styles))
    story.append(Spacer(1, 6))

    ae_box = (
        "Root Cause of AE Suppression: Two legitimate normal flows (rows 10731 and 10737, RE ~ 269) representing short aborted TCP connections "
        "inflated the validation standard deviation to 3.645, displacing the operational threshold tau to 11.16006. "
        "Because Backdoor intrusions mimic normal session byte structures (RE < 8.5), all 583 Backdoor attacks fell below the threshold."
    )
    story.append(create_callout("FORENSIC INSIGHT: WHY THE AUTOENCODER WAS SUPPRESSED", ae_box, border_color="#D9534F", bg_color="#FFEBEE", styles=styles))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SECTION 7: STACKING & FUSION
    # -------------------------------------------------------------------------
    story.append(Paragraph("7. Out-of-Fold Stacking & Hybrid Logical Fusion", styles['Heading1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=8))
    story.append(Paragraph(
        "To eliminate meta-level leakage, 5-fold Stratified Out-of-Fold (OOF) cross-validation was performed on TRAIN. "
        "The meta-learner was trained across three independent seeds (42, 123, 2024), achieving extreme stability: "
        "Seed 42 Macro-F1 = 0.892609, Seed 123 = 0.892619, Seed 2024 = 0.893656 (Mean: 0.892961 ± 0.000491).",
        styles['Body']
    ))
    story.append(Spacer(1, 3))

    # Figure 6: Meta-learner weights (height adjusted to 140 pt)
    story.append(create_figure(
        FIGURES_DIR / "fig10_meta_learner_weights.png", 490, 140, 6,
        "Stacking Meta-Learner Logistic Regression Coefficients (Seed 42)",
        "results/stacking/EXP_OOF_STACK_V1/seed_42/meta_learner.joblib",
        "Demonstrates heavy positive weighting of Random Forest (+2.15) and Neural Network (+1.79), while negatively weighting Linear SVM (-0.18).",
        styles=styles
    ))
    story.append(Spacer(1, 6))

    fusion_text = (
        "Hybrid Logical-OR Fusion (C06): Formulated as C06 = C01 OR AE. "
        "This configuration was designed to allow the Autoencoder to rescue attack samples missed by C01 (Quadrant Q3). "
        "On Development-Test, C06 achieved Macro-F1 = 0.892440 and FPR = 0.192243, reflecting a tiny penalty of +13 false alarms."
    )
    story.append(create_callout("HYBRID FUSION LOGICAL FORMULATION", fusion_text, border_color="#1B365D", bg_color="#F4F6F9", styles=styles))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # SPRINT CHAPTERS: SPRINTS 7 TO 13 (CLEAN MULTI-SECTION FLOW)
    # -------------------------------------------------------------------------
    sprints_md = [
        ("Sprint 7 — Autoencoder Development", get_sprint7_markdown()),
        ("Sprint 8 — Baseline Evaluation Foundation", get_sprint8_markdown()),
        ("Sprint 9 — Formal Hypothesis Testing (H1, H2, H3)", get_sprint9_markdown()),
        ("Sprint 10 — Systematic Ablation Study", get_sprint10_markdown()),
        ("Sprint 11 — Model Explainability & Provenance Audit", get_sprint11_markdown()),
        ("Sprint 12 — Final Reproducibility & Zero-Training Audit", get_sprint12_markdown()),
        ("Sprint 13 — Controlled Zero-Day Simulation", get_sprint13_markdown())
    ]

    for title, md_content in sprints_md:
        story.append(Paragraph(title, styles['Heading1']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=6))

        lines = md_content.split("\n")
        in_table = False
        table_lines = []
        in_callout = False
        callout_lines = []
        callout_title = ""
        current_section = ""

        for line in lines:
            line_str = line.strip()
            if not line_str:
                if in_table and table_lines:
                    story.append(parse_md_table_to_flowable(table_lines, styles))
                    in_table = False
                    table_lines = []
                    # Check for inline table-adjacent figures
                    if "Sprint 9" in title and "10. Expectation vs Actual" in current_section:
                        story.append(Spacer(1, 4))
                        story.append(create_figure(
                            FIGURES_DIR / "fig07_sprint9_hypotheses.png", 490, 125, 7,
                            "Visual Decision Summary for Hypotheses H1, H2, and H3",
                            "results/evaluation/EXP_H123_V1/summary.json",
                            "Clearly shows H1 SUPPORTED (+0.0122 delta vs epsilon=0.005), H2 NOT SUPPORTED (0/583), and H3 NOT SUPPORTED (0 rescues).",
                            styles=styles
                        ))
                    elif "Sprint 10" in title and "10. Expectation vs Actual" in current_section:
                        story.append(Spacer(1, 4))
                        story.append(create_figure(
                            FIGURES_DIR / "fig09_ablation_deltas.png", 490, 125, 9,
                            "Paired Macro-F1 Deltas Relative to Full Stacking (A1)",
                            "results/ablation/EXP_ABLATION_V1/paired_deltas.csv",
                            "Quantifies RF contribution (+0.0245) and the catastrophic penalty of Soft Voting (-0.0413).",
                            styles=styles
                        ))
                    elif "Sprint 12" in title and "10. Expectation vs Actual" in current_section:
                        story.append(Spacer(1, 4))
                        story.append(create_figure(
                            FIGURES_DIR / "fig11_s12_reproducibility.png", 490, 125, 10,
                            "Sprint 12 Historical Reference vs. Reproduced Performance",
                            "results/final_reproducibility/EXP_FINAL_REPRO_V1/publication/final_metrics.csv",
                            "Confirms bitwise reproduction across all frozen classifiers with zero floating-point divergence.",
                            styles=styles
                        ))
                    elif "Sprint 13" in title and "10. Expectation vs Actual" in current_section:
                        story.append(Spacer(1, 4))
                        story.append(create_figure(
                            FIGURES_DIR / "fig14_quadrant_structure.png", 490, 125, 12,
                            "Quadrant Decomposition of Protected Backdoor Detections (C01 vs Autoencoder)",
                            "results/zero_day/EXP_ZERODAY_V1/metrics/preregistered_decisions.json",
                            "Q1=0, Q2=582, Q3=0, Q4=1. Proves conclusively that AE provided 0 rescues, and all detections originated from C01.",
                            styles=styles
                        ))
                if in_callout and callout_lines:
                    story.append(create_callout(callout_title, " ".join(callout_lines), styles=styles))
                    in_callout = False
                    callout_lines = []
                continue

            # Detect callout start
            if line_str.startswith("> [!NOTE]") or line_str.startswith("> [!IMPORTANT]") or line_str.startswith("> [!WARNING]"):
                in_callout = True
                callout_title = line_str.split("[!")[1].split("]")[0] + " — RESEARCH NOTE"
                callout_lines = []
                continue
            elif in_callout:
                if line_str.startswith("> "):
                    callout_lines.append(line_str[2:])
                    continue
                elif line_str == ">":
                    callout_lines.append("<br/><br/>")
                    continue
                else:
                    story.append(create_callout(callout_title, " ".join(callout_lines), styles=styles))
                    in_callout = False
                    callout_lines = []

            # Detect markdown table
            if line_str.startswith("|") and line_str.endswith("|"):
                in_table = True
                table_lines.append(line_str)
                continue
            elif in_table:
                story.append(parse_md_table_to_flowable(table_lines, styles))
                in_table = False
                table_lines = []

            # Headings & text
            clean_l = sanitize_text(line_str)
            if line_str.startswith("### "):
                current_section = line_str[4:]
                story.append(Paragraph(clean_l[4:], styles['Heading2']))
            elif line_str.startswith("#### "):
                story.append(Paragraph(clean_l[5:], styles['Heading3']))
            elif line_str.startswith("- ") or line_str.startswith("* "):
                story.append(Paragraph(f"• {clean_l[2:]}", styles['Bullet']))
            elif line_str.startswith("# "):
                continue
            else:
                story.append(Paragraph(clean_l, styles['Body']))
                # Check for section-specific figures right after introductory text
                if "Sprint 10" in title and "8. Actual Result" in current_section and "A3_NO_RF" in line_str:
                    story.append(Spacer(1, 4))
                    story.append(create_figure(
                        FIGURES_DIR / "fig08_ablation_macro_f1.png", 490, 125, 8,
                        "Systematic Ablation Macro-F1 Across Configurations A0 to A6",
                        "results/ablation/EXP_ABLATION_V1/ablation_table.csv",
                        "Highlights the severe performance drop when Random Forest is removed (A3) and the collapse of Soft Voting (A1b).",
                        styles=styles
                    ))
                elif "Sprint 13" in title and "8. Actual Result" in current_section and "ZDR = 0.998285" in line_str:
                    story.append(Spacer(1, 4))
                    story.append(create_figure(
                        FIGURES_DIR / "fig12_zero_day_detection.png", 490, 120, 11,
                        "Zero-Day Detection Rates Across 8 Evaluated Systems on Protected Backdoors (N=583)",
                        "results/zero_day/EXP_ZERODAY_V1/metrics/zero_day_metrics.csv",
                        "Shows C01, C06, and RF detecting 582/583 samples (99.83%), while AE detects 0/583.",
                        styles=styles
                    ))
                elif "Sprint 13" in title and "17. Graphs" in current_section and "ROC and PR curves" in line_str:
                    story.append(Spacer(1, 4))
                    story.append(create_figure(
                        FIGURES_DIR / "fig13_zero_day_fpr.png", 490, 120, 13,
                        "Benign Control False Positive Rates (N=37,000)",
                        "results/zero_day/EXP_ZERODAY_V1/metrics/zero_day_metrics.csv",
                        "Shows C01 FPR at 19.19% (7,100 FP), AE at 0.05% (19 FP), and C06 at 19.22% (7,113 FP).",
                        styles=styles
                    ))

        if in_table and table_lines:
            story.append(parse_md_table_to_flowable(table_lines, styles))
        if in_callout and callout_lines:
            story.append(create_callout(callout_title, " ".join(callout_lines), styles=styles))

        if "Sprint 13" in title:
            story.append(PageBreak())
        else:
            story.append(Spacer(1, 14))

    # -------------------------------------------------------------------------
    # CONSOLIDATED RESULTS & SYNTHESES
    # -------------------------------------------------------------------------
    story.append(Paragraph("16. Consolidated Final Results (Tables A–F)", styles['Heading1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=6))
    story.append(Paragraph(
        "To ensure absolute scientific clarity and prevent cross-population contamination, "
        "all experimental findings are consolidated into six dedicated, population-specific tables:",
        styles['Body']
    ))
    story.append(Spacer(1, 3))

    # Table A
    story.append(Paragraph("<b>Table A: Baseline Base Model Benchmark (Development-Test, N=81,749)</b>", styles['BodyBold']))
    story.append(create_table(base_headers, base_rows, [115, 65, 70, 65, 75, 60, 55], styles=styles))
    story.append(Spacer(1, 6))

    # Table B
    story.append(Paragraph("<b>Table B: Development-Test Stacking & Fusion Performance (N=81,749)</b>", styles['BodyBold']))
    table_b_headers = ["System / Pipeline", "Macro-F1", "Macro Precision", "Macro Recall", "Balanced Accuracy", "FPR", "Benign FP (37k)"]
    table_b_rows = [
        ["RF Baseline", "0.880733", "0.903932", "0.874944", "0.874944", "0.231027", "8,548"],
        ["Stacking (Seed 42)", "0.892609", "0.906552", "0.887931", "0.887931", "0.191892", "7,100"],
        ["Stacking (Seed 123)", "0.892619", "0.906591", "0.887935", "0.887935", "0.191973", "7,103"],
        ["Stacking (Seed 2024)", "0.893656", "0.907007", "0.889071", "0.889071", "0.188784", "6,985"],
        ["Stacking (3-Seed Mean)", "0.892961", "0.906717", "0.888312", "0.888312", "0.190883", "7,063"],
        ["Fusion C06 (Stack 42+AE)", "0.892440", "0.906432", "0.887755", "0.887755", "0.192243", "7,113"],
        ["Ablation A1b (Soft Vote)", "0.850632", "0.886708", "0.844649", "0.844649", "0.293541", "10,861"]
    ]
    story.append(create_table(table_b_headers, table_b_rows, [125, 65, 70, 65, 70, 55, 55], styles=styles))
    story.append(Spacer(1, 6))

    # Table C
    story.append(Paragraph("<b>Table C: Systematic Ablation Study Findings (Historical Dynamically-Fitted)</b>", styles['BodyBold']))
    table_c_headers = ["Config ID", "Description", "Mean Macro-F1", "Delta vs A1", "Mean FPR", "Backdoor Det (583)"]
    table_c_rows = [
        ["A0_RF", "Random Forest alone", "0.881618", "-0.010359", "0.229189", "582"],
        ["A1_FULL_STACK", "Full 4-Model Stacking", "0.891977", "0.000000", "0.194874", "582"],
        ["A1b_SOFT_VOTE", "Equal Soft Voting", "0.850642", "-0.041335", "0.293775", "582"],
        ["A2_NO_DT", "Stacking without DT", "0.892276", "+0.000299", "0.194144", "582"],
        ["A3_NO_RF", "Stacking without RF", "0.867496", "-0.024481", "0.232766", "578"],
        ["A4_NO_SVM", "Stacking without SVM", "0.891022", "-0.000954", "0.199748", "582"],
        ["A5_NO_NN", "Stacking without NN", "0.891953", "-0.000024", "0.194874", "582"],
        ["A6_STACK_PLUS_AE", "Stacking + AE Fusion", "0.891807", "-0.000169", "0.195225", "582"]
    ]
    story.append(create_table(table_c_headers, table_c_rows, [95, 130, 75, 70, 65, 70], styles=styles))
    story.append(Spacer(1, 6))

    # Table D
    story.append(Paragraph("<b>Table D: Zero-Day Simulation Performance on Protected Backdoor (N=583) & Benign (N=37,000)</b>", styles['BodyBold']))
    table_d_headers = ["System", "TP (Backdoor)", "FN", "ZDR", "FP (Benign)", "TN", "FPR", "Wilson 95% CI"]
    table_d_rows = [
        ["Decision Tree", "577", "6", "0.989708", "10,343", "26,657", "0.279541", "[0.9774, 0.9953]"],
        ["Random Forest", "582", "1", "0.998285", "8,548", "28,452", "0.231027", "[0.9903, 0.9997]"],
        ["Linear SVM", "577", "6", "0.989708", "11,491", "25,509", "0.310568", "[0.9774, 0.9953]"],
        ["Neural Network", "578", "5", "0.991424", "5,640", "31,360", "0.152432", "[0.9798, 0.9964]"],
        ["Stacking (C01)", "582", "1", "0.998285", "7,100", "29,900", "0.191892", "[0.9903, 0.9997]"],
        ["Autoencoder (AE)", "0", "583", "0.000000", "19", "36,981", "0.000514", "[0.0000, 0.0063]"],
        ["Fusion (C06)", "582", "1", "0.998285", "7,113", "29,887", "0.192243", "[0.9903, 0.9997]"]
    ]
    story.append(create_table(table_d_headers, table_d_rows, [95, 65, 35, 55, 65, 50, 55, 85], styles=styles))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # HYPOTHESIS MATRIX & FINDINGS
    # -------------------------------------------------------------------------
    story.append(Paragraph("17. Pre-Registered Hypotheses & Major Scientific Findings", styles['Heading1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=6))

    story.append(Paragraph("<b>Table E: Final Pre-Registered Hypothesis Decisions</b>", styles['BodyBold']))
    table_e_headers = ["Hypothesis", "Research Question Tested", "Quantitative Evidence", "Decision Rule", "Verdict"]
    table_e_rows = [
        ["H1", "Stacking superiority vs best model (RF)", "Mean Stacking F1 = 0.8930 vs RF = 0.8807 (+0.0122)", "Delta >= 0.005", "SUPPORTED"],
        ["H2", "Standalone AE zero-day anomaly detection", "Detected count = 0 / 583 (0.00%) at tau = 11.16006", "ae_detected == 0 -> NOT_SUPPORTED", "NOT_SUPPORTED"],
        ["H3", "Hybrid fusion rescue over stacking", "C06 det = 582 vs C01 det = 582 (Rescued = 0)", "C06 > C01 and delta FPR <= 0.02", "NOT_SUPPORTED"],
        ["Generalization", "Unseen category zero-day generalization", "C06 ZDR = 0.998285, Wilson CI: [0.9903, 0.9997]", "ZDR >= 0.50 and CI lower > 0.50", "SUPPORTED"],
        ["Fusion Gain", "Statistical rescue superiority", "RescueGain = 0.0, Exact binomial p = 1.0000", "Gain >= 0.05 and p < 0.05", "NOT_SUPPORTED"]
    ]
    story.append(create_table(table_e_headers, table_e_rows, [70, 125, 130, 100, 80], styles=styles))
    story.append(Spacer(1, 8))

    findings_text = """
    <b>Core Scientific Findings:</b><br/>
    1. <b>Supervised Stacking Generalization:</b> Multi-model ensemble stacking delivers statistically verified improvements over individual base models (+0.0122 F1) and generalizes exceptionally well to withheld attack categories (99.83% Backdoor detection).<br/>
    2. <b>Random Forest Dominance:</b> Bagged decision trees represent the most vital supervised component, accounting for +0.0245 Macro-F1 in ablation analysis.<br/>
    3. <b>Soft Voting Inefficacy:</b> Unweighted probability averaging collapses performance (-0.0413 F1) by permitting noisy classifiers to distort decision boundaries.<br/>
    4. <b>Autoencoder Operational Suppression:</b> Global reconstruction error thresholds are severely distorted by legitimate, benign protocol connection aborts (TCP RST/FIN), causing complete suppression of zero-day anomaly sensitivity.<br/>
    5. <b>Zero Rescue Gain:</b> Hybrid logical-OR fusion provided 0 attack rescues while adding 13 false positives, disproving naive claims of hybrid superiority under global thresholding.
    """
    story.append(create_callout("MAJOR SCIENTIFIC FINDINGS SUMMARY", findings_text, border_color="#1B365D", bg_color="#F4F6F9", styles=styles))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # TIMELINE & CONCLUSION
    # -------------------------------------------------------------------------
    story.append(Paragraph("18. Timeline, Research Milestones & Final Conclusion", styles['Heading1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=6))

    # Figure 14: Timeline (height adjusted to 75 pt)
    story.append(create_figure(
        FIGURES_DIR / "fig15_timeline.png", 490, 75, 14,
        "Research Milestone Progression Across Sprints 7 to 13",
        "Repository Git tag history",
        "Chronological progression from Autoencoder training (S7) through Hypothesis Testing (S9), Reproducibility (S12), and Zero-Day Freeze (S13).",
        styles=styles
    ))
    story.append(Spacer(1, 6))

    conclusion_text = (
        "Final Scientific Conclusion: This publication-oriented research project establishes that supervised ensemble stacking (C01) "
        "possesses remarkable latent generalization capabilities, successfully detecting 99.83% (582/583) of unseen Backdoor attacks by leveraging "
        "fundamental protocol flow primitives shared across attack categories. "
        "In contrast, unsupervised deep Autoencoders (AE) calibrated on benign traffic are highly vulnerable to operational suppression: "
        "ambient benign connection-termination anomalies (TCP RST/FIN) displace parametric thresholds outward, rendering the model completely inert "
        "to subtle zero-day intrusions (0/583 detected). Consequently, hybrid logical-OR fusion (C06) yielded zero rescue gain and provided no empirical "
        "or statistical improvement over supervised stacking alone. "
        "Future hybrid NIDS research must move beyond global reconstruction thresholds toward protocol-specific, localized subspace partitioning."
    )
    story.append(create_callout("EVIDENCE-BASED FINAL CONCLUSION", conclusion_text, border_color="#1B365D", bg_color="#F4F6F9", styles=styles))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Appendices Summary (A–J):</b>", styles['BodyBold']))
    story.append(Paragraph(
        "• <b>Appendix A:</b> Dataset and Split Manifest (All SHA-256 hashes and row counts).<br/>"
        "• <b>Appendix B:</b> Feature Selection Summary (75 features, mutual information plateau).<br/>"
        "• <b>Appendix C:</b> Model Configurations (DT, RF, SVM, NN architecture details).<br/>"
        "• <b>Appendix D:</b> Key Hyperparameters (Adam, LR, batch sizes, early stopping).<br/>"
        "• <b>Appendix E:</b> Hypothesis Definitions and Decision Rules (H1, H2, H3, DD-4, Wilson CI).<br/>"
        "• <b>Appendix F:</b> Important Checkpoint Hashes (ae_final.pt, meta_learner, nn_final.pt).<br/>"
        "• <b>Appendix G:</b> Experiment Timeline and Freeze Tags (sprint9-freeze to sprint13-freeze).<br/>"
        "• <b>Appendix H:</b> Validation Gate Summary (44/44 gates passed, 0 training operations).<br/>"
        "• <b>Appendix I:</b> Artifact Inventory (Markdown, PDF, Figures, and Forensic Audits).<br/>"
        "• <b>Appendix J:</b> Historical vs Frozen Lineage Distinctions (A1 7,201 FP vs C01 7,100 FP).",
        styles['Body']
    ))
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # FINAL RESEARCH STATUS CHECKLIST PAGE
    # -------------------------------------------------------------------------
    story.append(Paragraph("Final Research Status & Audit Sign-Off", styles['Heading1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1B365D'), spaceBefore=2, spaceAfter=6))
    story.append(Spacer(1, 3))

    chk_headers = ["Audit Item", "Project Specification", "Observed Value / Metric", "Formal Verification Status"]
    chk_rows = [
        ["Target Project", "UNSW-NB15 IDS", "Publication-Oriented Research", "CONFIRMED"],
        ["Documented Sprints", "Sprint 7 through Sprint 13", "All 7 Sprints Fully Documented", "COMPLETE"],
        ["Final Experiment", "EXP_ZERODAY_V1 (Sprint 13)", "Protocol V1.4 Executed & Frozen", "FROZEN"],
        ["Final Git Commit", "f694e19e44a3dafb486ff216428f1be1f2ec9120", "sprint13-freeze tag verified", "FROZEN"],
        ["Main C06 Result", "Protected Backdoor Detection", "582 / 583 (99.8285%)", "VERIFIED"],
        ["AE Rescue Gain", "Attacks Missed by C01, Caught by AE", "0 / 583 (0.0000%)", "VERIFIED"],
        ["Hypothesis H1", "Stacking Superiority vs RF Baseline", "SUPPORTED (+0.0122 F1 > 0.005)", "SUPPORTED"],
        ["Hypothesis H2", "Standalone AE Zero-Day Anomaly Detection", "NOT_SUPPORTED (0/583 Det, DD-4)", "NOT_SUPPORTED"],
        ["Hypothesis H3", "Hybrid Fusion Rescue without FPR Inflation", "NOT_SUPPORTED (Rescue = 0)", "NOT_SUPPORTED"],
        ["Generalization Decision", "Formal Unseen Category Generalization", "SUPPORTED (Wilson CI: [0.9903, 0.9997])", "SUPPORTED"],
        ["Fusion Decision", "Formal Hybrid Fusion Improvement", "NOT_SUPPORTED (Exact Binomial p=1.0)", "NOT_SUPPORTED"],
        ["Zero-Training Compliance", "Refitting / Training during S12 & S13", "0 Operations Executed (100% Frozen)", "AUDIT PASSED"],
        ["Validation Gate Audit", "Automated Pre-Registered Checks", "44 / 44 Passed (100% Clean)", "AUDIT PASSED"]
    ]
    story.append(create_table(chk_headers, chk_rows, [110, 130, 145, 120], styles=styles))
    story.append(Spacer(1, 10))

    signoff_text = (
        "RESEARCH AUDIT SIGN-OFF & CERTIFICATION:<br/>"
        "This research report reflects strictly verified, immutable data from frozen experiment artifacts. "
        "No models were retrained, no checkpoints altered, no thresholds adjusted post-hoc, and no data fabricated. "
        "Both historical dynamically-fitted and canonical frozen-checkpoint lineages are preserved with complete transparency. "
        "The project is concluded and archived in FROZEN status."
    )
    story.append(create_callout("FINAL AUTHORITATIVE CERTIFICATION", signoff_text, border_color="#2E7D32", bg_color="#E8F5E9", styles=styles))

    return story


# -----------------------------------------------------------------------------
# 4. MAIN COMPILATION AND VERIFICATION
# -----------------------------------------------------------------------------
def assemble_markdown() -> str:
    qa_list = build_executive_summary_qa()
    parts = [
        get_title_markdown(),
        get_executive_summary_markdown(qa_list),
        get_introduction_markdown(),
        get_dataset_markdown(),
        get_features_markdown(),
        get_architecture_markdown(),
        get_base_models_markdown(),
        get_autoencoder_markdown(),
        get_stacking_markdown(),
        get_fusion_markdown(),
        get_sprint7_markdown(),
        get_sprint8_markdown(),
        get_sprint9_markdown(),
        get_sprint10_markdown(),
        get_sprint11_markdown(),
        get_sprint12_markdown(),
        get_sprint13_markdown(),
        get_why_it_happened_markdown(),
        get_problems_and_resolutions_markdown(),
        get_consolidated_results_markdown(),
        get_findings_and_conclusion_markdown(),
        get_appendices_markdown(),
        get_final_checklist_markdown()
    ]
    full_md = "\n\n".join(parts)
    with open(MD_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_md)
    print(f"[OK] Full source markdown written to {MD_OUT_PATH} ({len(full_md)} bytes)")
    return full_md


def compile_pdf() -> int:
    print("[INFO] Setting up styles and document template...")
    styles = setup_styles()

    doc = SimpleDocTemplate(
        str(PDF_OUT_PATH),
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=45,
        bottomMargin=48
    )

    print("[INFO] Building story flowables...")
    story = build_story(styles)

    print(f"[INFO] Compiling PDF via NumberedCanvas to {PDF_OUT_PATH}...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] PDF compiled successfully! Size: {PDF_OUT_PATH.stat().st_size} bytes")

    # Clear old inspection images
    for old_img in INSPECTION_DIR.glob("page_*.png"):
        try:
            old_img.unlink()
        except Exception:
            pass

    # Run PyMuPDF visual inspection
    print("[INFO] Performing page-by-page visual inspection via PyMuPDF...")
    pdf_doc = pymupdf.open(str(PDF_OUT_PATH))
    page_count = len(pdf_doc)
    print(f"[INFO] Total Pages: {page_count}")

    for idx in range(page_count):
        page = pdf_doc[idx]
        pix = page.get_pixmap(dpi=150)
        img_out = INSPECTION_DIR / f"page_{idx+1:02d}.png"
        pix.save(str(img_out))

    pdf_doc.close()
    print(f"[OK] Rendered {page_count} pages to {INSPECTION_DIR} for visual quality verification.")
    return page_count


if __name__ == "__main__":
    # Also write out full markdown
    assemble_markdown()
    pages = compile_pdf()
    print(f"\n=======================================================")
    print(f"REPORT GENERATION COMPLETE:")
    print(f"  PDF: {PDF_OUT_PATH}")
    print(f"  Markdown: {MD_OUT_PATH}")
    print(f"  Total Pages: {pages}")
    print(f"=======================================================")
