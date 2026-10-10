"""Render reports/REPORT.md to a self-contained HTML file and a print-ready PDF.

Images are inlined as base64 so both outputs are single files that can be
emailed or uploaded with nothing else attached. The PDF is produced by printing
the HTML in headless Chrome/Edge, which is the only renderer on this machine
that handles the tables and figures properly.
"""
import base64
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MD = REPORTS / "REPORT.md"
HTML = REPORTS / "Week1_Preprocessing_Report.html"
PDF = REPORTS / "Week1_Preprocessing_Report.pdf"

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
:root {
  --ink: #15140f; --ink-2: #44423c; --muted: #76736a;
  --rule: #e0ded7; --surface: #ffffff; --accent: #2a78d6; --soft: #f7f6f3;
}
* { box-sizing: border-box; }
body {
  font-family: "Charter", "Georgia", "Cambria", serif;
  color: var(--ink); background: var(--surface);
  font-size: 10.6pt; line-height: 1.58; margin: 0 auto; max-width: 196mm;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 23pt; line-height: 1.2; margin: 0 0 4mm; letter-spacing: -0.01em;
}
h2 {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 14pt; margin: 9mm 0 3mm; padding-bottom: 1.6mm;
  border-bottom: 1.5px solid var(--rule); break-after: avoid;
}
h3 {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 11.4pt; margin: 6mm 0 2mm; color: var(--ink); break-after: avoid;
}
p, li { orphans: 3; widows: 3; }
p { margin: 0 0 2.6mm; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1.2mm; }
strong { font-weight: 700; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 7mm 0; }
a { color: var(--accent); text-decoration: none; }
code {
  font-family: "Cascadia Mono", "Consolas", monospace; font-size: 8.9pt;
  background: var(--soft); padding: 0.5mm 1.1mm; border-radius: 2px;
  border: 1px solid var(--rule);
}
pre {
  background: var(--soft); border: 1px solid var(--rule); border-left: 3px solid var(--accent);
  padding: 3mm 4mm; border-radius: 3px; overflow-x: auto; break-inside: avoid;
  font-size: 8.9pt; line-height: 1.45; margin: 0 0 3.5mm;
}
pre code { background: none; border: 0; padding: 0; font-size: inherit; }
table {
  border-collapse: collapse; width: 100%; margin: 0 0 4mm;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 8.9pt; break-inside: avoid;
}
th {
  text-align: left; background: var(--soft); color: var(--ink);
  font-weight: 600; padding: 1.9mm 2.4mm; border-bottom: 1.5px solid var(--rule);
}
td { padding: 1.7mm 2.4mm; border-bottom: 1px solid var(--rule); color: var(--ink-2); }
tr:last-child td { border-bottom: 0; }
td:first-child, th:first-child { padding-left: 0.5mm; }
img {
  max-width: 100%; height: auto; display: block; margin: 3mm auto 5mm;
  break-inside: avoid; border: 1px solid var(--rule); border-radius: 3px;
}
blockquote {
  margin: 0 0 3.5mm; padding: 2.5mm 4mm; background: var(--soft);
  border-left: 3px solid var(--accent); border-radius: 0 3px 3px 0; color: var(--ink-2);
}
blockquote p:last-child { margin-bottom: 0; }
.subtitle {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  color: var(--muted); font-size: 9.6pt; margin: 0 0 7mm;
  padding-bottom: 4mm; border-bottom: 2px solid var(--ink);
}
h2 + p > img:first-child { margin-top: 1mm; }
"""


def inline_images(html: str) -> str:
    """Replace <img src="figures/x.png"> with a base64 data URI."""
    def repl(m):
        src = m.group(1)
        path = (REPORTS / src).resolve()
        if not path.exists():
            print(f"  WARNING: missing image {src}")
            return m.group(0)
        b64 = base64.b64encode(path.read_bytes()).decode()
        return m.group(0).replace(src, f"data:image/png;base64,{b64}")
    return re.sub(r'<img[^>]+src="([^"]+)"', repl, html)


def main() -> int:
    text = MD.read_text(encoding="utf-8")

    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "codehilite", "attr_list", "sane_lists"],
        extension_configs={"codehilite": {"noclasses": True, "pygments_style": "friendly"}},
    )
    body = inline_images(body)

    # Promote the first paragraph after the H1 into a styled subtitle block.
    body = body.replace(
        "<p><strong>Deliverable:</strong>", '<p class="subtitle"><strong>Deliverable:</strong>', 1)

    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>Week 1 - Python for Machine Learning &amp; Data Preprocessing</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    HTML.write_text(html, encoding="utf-8")
    print(f"HTML -> {HTML}  ({HTML.stat().st_size/1024:.0f} KB, images inlined)")

    browser = next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("chrome")
    if not browser:
        print("No Chrome/Edge found - HTML written, but PDF skipped.")
        print("Open the HTML in any browser and use Print > Save as PDF.")
        return 1

    if PDF.exists():
        PDF.unlink()
    cmd = [
        browser, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", "--virtual-time-budget=20000",
        f"--print-to-pdf={PDF}", HTML.as_uri(),
    ]
    print(f"printing with {Path(browser).name} ...")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if not PDF.exists():
        print("PDF generation failed:\n", proc.stderr[-1500:])
        return 1
    print(f"PDF  -> {PDF}  ({PDF.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
