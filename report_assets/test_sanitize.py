import re
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def sanitize_text(txt: str) -> str:
    # 1. LaTeX math substitutions
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

styles = getSampleStyleSheet()
test_str = 'Line 1<br/><br/>Line 2 with <b>bold</b> and <font color="#1B365D">blue</font> and <sub>sub</sub>.'
sanitized = sanitize_text(test_str)
p = Paragraph(sanitized, styles['Normal'])
print("SUCCESS: Paragraph with <br/> created!")
print(sanitized.encode('ascii', 'backslashreplace').decode())
p = Paragraph(sanitized, styles['Normal'])
print("SUCCESS: Paragraph created without XML parse errors!")
print(sanitized.encode('ascii', 'backslashreplace').decode())
