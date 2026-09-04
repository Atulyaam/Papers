#!/usr/bin/env python3
"""
scripts/build_research_report_pdf.py
------------------------------------
Master generator for the Complete UNSW-NB15 Research Report.
Produces:
1. UNSW_NB15_Complete_Research_Report.pdf (Print-ready, academic ReportLab PDF)
2. UNSW_NB15_Complete_Research_Report.md  (Complete source Markdown document)
3. Page-by-page visual inspection images in report_assets/inspections/

Strictly adheres to all scientific integrity rules, authoritative project data,
and reporting guidelines.
"""

import sys
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple

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

ROOT = Path(__file__).resolve().parent.parent
REPORT_PDF_PATH = ROOT / "UNSW_NB15_Complete_Research_Report.pdf"
REPORT_MD_PATH = ROOT / "UNSW_NB15_Complete_Research_Report.md"
ASSETS_DIR = ROOT / "report_assets"
FIGURES_DIR = ASSETS_DIR / "figures"
INSPECTION_DIR = ASSETS_DIR / "inspections"
INSPECTION_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# TWO-PASS NUMBERED CANVAS (RUNNING HEADERS & FOOTERS)
# -----------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and display total page count.
    Adds running headers and footers with academic styling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # Page 1 is the Title Page: omit running headers/footers
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#1B365D"))

        # Running Header
        header_text = "UNSW-NB15 INTRUSION DETECTION SYSTEM — COMPLETE RESEARCH REPORT"
        self.drawString(45, 842 - 32, header_text)
        self.setFont("Helvetica-Oblique", 7.5)
        self.setFillColor(colors.HexColor("#555555"))
        self.drawRightString(595 - 45, 842 - 32, "EXP_ZERODAY_V1 | SPRINT 7–13")

        # Top rule
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.6)
        self.line(45, 842 - 37, 595 - 45, 842 - 37)

        # Bottom rule
        self.line(45, 42, 595 - 45, 42)

        # Running Footer
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#666666"))
        self.drawString(45, 30, "CONFIDENTIAL & AUTHORITATIVE RESEARCH AUDIT — FROZEN BENCHMARK")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.setFont("Helvetica-Bold", 8)
        self.drawRightString(595 - 45, 30, page_str)

        self.restoreState()


print("NumberedCanvas initialized.")
