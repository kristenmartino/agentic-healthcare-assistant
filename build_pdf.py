"""Convert WRITEUP.md to WRITEUP.pdf via markdown → HTML → playwright print-to-PDF."""
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent
MD_PATH = ROOT / "WRITEUP.md"
HTML_PATH = ROOT / "WRITEUP.html"
PDF_PATH = ROOT / "WRITEUP.pdf"

CSS = """
body { font-family: -apple-system, 'Helvetica Neue', sans-serif; max-width: 780px;
       margin: 40px auto; line-height: 1.6; color: #222; padding: 0 20px; }
h1 { border-bottom: 2px solid #333; padding-bottom: 8px; margin-top: 0; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 32px; }
h3 { margin-top: 24px; color: #333; }
code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 90%; }
pre { background: #f7f7f7; padding: 12px; border-radius: 5px; overflow-x: auto;
      font-size: 85%; line-height: 1.4; }
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; margin: 12px 0; width: 100%; font-size: 92%; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; vertical-align: top; }
th { background: #f4f4f4; }
blockquote { border-left: 4px solid #888; padding: 8px 16px; color: #555;
             margin: 16px 0; background: #fafafa; }
a { color: #0366d6; text-decoration: none; }
a:hover { text-decoration: underline; }
"""

md_text = MD_PATH.read_text()
html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style><title>Healthcare Assistant — WRITEUP</title></head>
<body>{html_body}</body></html>"""
HTML_PATH.write_text(html)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_context().new_page()
    page.goto(HTML_PATH.as_uri(), wait_until="load")
    page.pdf(path=str(PDF_PATH), format="A4", margin={"top": "20mm", "bottom": "20mm",
                                                        "left": "16mm", "right": "16mm"},
             print_background=True)
    browser.close()

# Clean up the intermediate HTML
HTML_PATH.unlink()

size_kb = PDF_PATH.stat().st_size // 1024
print(f"✓ {PDF_PATH.name} ({size_kb} KB)")
