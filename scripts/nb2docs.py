#!/usr/bin/env python3
"""Convert an executed Jupyter notebook to a Starlight-compatible Markdown page.

Processes cell-by-cell:
  - Markdown cells → raw markdown
  - Code cells → ```python fenced blocks
  - Text/stream outputs → ```text blocks
  - HTML outputs (pandas tables) → raw HTML
  - Plotly outputs → standalone .html saved to plots dir, embedded via <iframe>
  - Image outputs (png) → saved to plots dir, embedded via ![alt](path)

Usage:
    python scripts/nb2docs.py \\
        --input notebooks/B1_interactive_explorer.ipynb \\
        --output docs/src/content/docs/tutorials/b1-explorer.md \\
        --plots-dir docs/public/plots/ \\
        --title "Interactive Explorer" \\
        --description "..."
"""

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path


PLOTLY_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { margin: 0; background: #1a1a2e; overflow: hidden; }
    #plot { width: 100%; height: 100%; position: absolute; top: 0; left: 0; }
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    var figure = {figure_json};
    // Override fixed dimensions — let iframe control size
    figure.layout.width = undefined;
    figure.layout.height = undefined;
    figure.layout.autosize = true;
    Plotly.newPlot('plot', figure.data, figure.layout, {responsive: true});
  </script>
</body>
</html>"""


def parse_args():
    p = argparse.ArgumentParser(description="Convert executed .ipynb to Starlight .md")
    p.add_argument("--input", required=True, help="Path to .ipynb file")
    p.add_argument("--output", required=True, help="Output .md file path")
    p.add_argument("--plots-dir", required=True, help="Directory for Plotly HTML and images")
    p.add_argument("--title", required=True, help="Page title (Starlight frontmatter)")
    p.add_argument("--description", default="", help="Page description")
    p.add_argument("--base-url", default="/chronos-vector/plots/", help="URL base for plot files")
    p.add_argument("--skip-code", action="store_true", help="Omit code cells (show only outputs)")
    p.add_argument("--dry-run", action="store_true", help="Print output without writing files")
    return p.parse_args()


def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cell_source(cell):
    """Get cell source as a single string."""
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src


def should_skip(cell):
    """Check if cell has nb2docs:skip metadata."""
    meta = cell.get("metadata", {})
    tags = meta.get("tags", [])
    if "nb2docs:skip" in tags:
        return True
    if meta.get("nb2docs") == "skip":
        return True
    return False


def extract_plotly_figure(output):
    """Extract Plotly figure JSON from a display_data output."""
    data = output.get("data", {})
    # Try plotly MIME type first
    for mime in ["application/vnd.plotly.v1+json", "application/vnd.plotly+json"]:
        if mime in data:
            return data[mime]
    return None


def save_plotly_html(figure_json, path):
    """Save a Plotly figure as standalone responsive HTML."""
    # Remove fixed dimensions so the plot fills the iframe
    layout = figure_json.get("layout", {})
    layout.pop("width", None)
    layout.pop("height", None)
    layout["autosize"] = True
    figure_json["layout"] = layout
    html = PLOTLY_HTML_TEMPLATE.replace("{figure_json}", json.dumps(figure_json))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def save_image(b64_data, path):
    """Save a base64-encoded image to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_data))


def process_outputs(outputs, slug, plots_dir, base_url, fig_counter):
    """Process cell outputs and return markdown + updated counter."""
    md_parts = []

    for output in outputs:
        otype = output.get("output_type", "")

        # --- Plotly figure ---
        plotly_fig = extract_plotly_figure(output)
        if plotly_fig is not None:
            fig_name = f"{slug}_fig_{fig_counter}.html"
            fig_path = os.path.join(plots_dir, fig_name)
            save_plotly_html(plotly_fig, fig_path)
            url = f"{base_url}{fig_name}"
            md_parts.append(
                f'\n<iframe src="{url}" width="100%" height="620" '
                f'style="border:none; border-radius:8px; margin:1rem 0;"></iframe>\n'
            )
            fig_counter += 1
            continue

        # --- Image (PNG) ---
        data = output.get("data", {})
        if "image/png" in data:
            img_name = f"{slug}_img_{fig_counter}.png"
            img_path = os.path.join(plots_dir, img_name)
            img_b64 = data["image/png"]
            if isinstance(img_b64, list):
                img_b64 = "".join(img_b64)
            save_image(img_b64, img_path)
            url = f"{base_url}{img_name}"
            md_parts.append(f"\n![Output]({url})\n")
            fig_counter += 1
            continue

        # --- HTML output (pandas tables) ---
        if "text/html" in data:
            html = data["text/html"]
            if isinstance(html, list):
                html = "".join(html)
            # Only include if it looks like a table (skip plotly widget HTML)
            if "<table" in html.lower():
                md_parts.append(f'\n<div class="nb-output">\n{html}\n</div>\n')
                continue

        # --- Text/stream output ---
        text = None
        if otype == "stream":
            text = output.get("text", "")
        elif otype in ("execute_result", "display_data"):
            text = data.get("text/plain", "")

        if text:
            if isinstance(text, list):
                text = "".join(text)
            text = text.rstrip()
            if text:
                md_parts.append(f"\n```text\n{text}\n```\n")

        # --- Error output ---
        if otype == "error":
            tb = output.get("traceback", [])
            # Strip ANSI escape codes
            ansi_re = re.compile(r"\x1b\[[0-9;]*m")
            tb_clean = "\n".join(ansi_re.sub("", line) for line in tb)
            if tb_clean.strip():
                md_parts.append(f"\n```text\n{tb_clean}\n```\n")

    return "".join(md_parts), fig_counter


def convert_notebook(nb, args):
    """Convert notebook dict to Starlight markdown string."""
    slug = Path(args.output).stem
    lines = []

    # Frontmatter
    lines.append("---")
    lines.append(f'title: "{args.title}"')
    if args.description:
        lines.append(f'description: "{args.description}"')
    lines.append("---\n")

    fig_counter = 0
    first_heading_stripped = False

    for cell in nb.get("cells", []):
        if should_skip(cell):
            continue

        ctype = cell.get("cell_type", "")
        source = cell_source(cell)

        # --- Markdown cell ---
        if ctype == "markdown":
            content = source.strip()
            # Strip the first H1 heading (Starlight generates it from title)
            if not first_heading_stripped and content.startswith("# "):
                first_nl = content.find("\n")
                if first_nl > 0:
                    content = content[first_nl:].strip()
                else:
                    content = ""
                first_heading_stripped = True
            if content:
                lines.append(content)
                lines.append("")
            continue

        # --- Code cell ---
        if ctype == "code":
            source = source.strip()
            if not source:
                continue

            if not args.skip_code:
                lines.append("```python")
                lines.append(source)
                lines.append("```\n")

            # Process outputs
            outputs = cell.get("outputs", [])
            if outputs:
                out_md, fig_counter = process_outputs(
                    outputs, slug, args.plots_dir, args.base_url, fig_counter
                )
                if out_md.strip():
                    lines.append(out_md)

    return "\n".join(lines)


def main():
    args = parse_args()

    nb = load_notebook(args.input)
    md = convert_notebook(nb, args)

    if args.dry_run:
        print(md)
        print(f"\n--- DRY RUN: would write to {args.output} ---")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)

    # Count generated plots
    slug = Path(args.output).stem
    if os.path.isdir(args.plots_dir):
        plots = [f for f in os.listdir(args.plots_dir) if f.startswith(slug)]
        print(f"Generated: {args.output} ({len(plots)} plots)")
    else:
        print(f"Generated: {args.output}")


if __name__ == "__main__":
    main()
