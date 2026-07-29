from __future__ import annotations

import re
import tempfile
from copy import copy
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PageObject, PdfReader, PdfWriter, Transformation


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "gnn_bandgap_research_final_slides.tex"
SLIDES_PDF = ROOT / "gnn_bandgap_research_final_slides.pdf"
OUT_PDF = ROOT / "gnn_bandgap_research_final_slides_pympress.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def extract_braced_commands(text: str, command: str) -> list[str]:
    marker = "\\" + command + "{"
    values: list[str] = []
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx == -1:
            return values
        pos = idx + len(marker)
        depth = 1
        buf: list[str] = []
        while pos < len(text) and depth:
            ch = text[pos]
            if ch == "\\":
                if pos + 1 < len(text):
                    buf.append(ch)
                    pos += 1
                    buf.append(text[pos])
                    pos += 1
                    continue
            if ch == "{":
                depth += 1
                buf.append(ch)
            elif ch == "}":
                depth -= 1
                if depth:
                    buf.append(ch)
            else:
                buf.append(ch)
            pos += 1
        values.append("".join(buf).strip())
        start = pos


def extract_frame_titles(text: str) -> list[str]:
    titles: list[str] = []
    pattern = re.compile(r"\\begin\{frame\}(?:\{([^}]*)\})?")
    for match in pattern.finditer(text):
        title = match.group(1)
        if title is None:
            # The first titlepage frame has no explicit title.
            titles.append("标题页")
        else:
            titles.append(clean_latex(title))
    return titles


def clean_latex(text: str) -> str:
    replacements = {
        r"\_": "_",
        r"\%": "%",
        r"\&": "&",
        r"--": "-",
        r"~": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\texttt\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\best\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            lines.append("")
            continue
        current = ""
        # Chinese text has no spaces, so wrap by character while preserving Latin spaces.
        tokens = re.findall(r"[A-Za-z0-9_./+\-:]+|\s+|.", para)
        for token in tokens:
            candidate = current + token
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= width or not current:
                current = candidate
            else:
                lines.append(current.rstrip())
                current = token.lstrip()
        if current:
            lines.append(current.rstrip())
    return lines


def make_note_pdf(note: str, title: str, index: int, total: int, path: Path) -> None:
    img_w, img_h = 1600, 900
    margin_x = 74
    image = Image.new("RGB", (img_w, img_h), "#e6e6e6")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(str(FONT_PATH), 38)
    meta_font = ImageFont.truetype(str(FONT_PATH), 24)
    body_font = ImageFont.truetype(str(FONT_PATH), 36)

    draw.rectangle((0, 0, img_w, 118), fill="#d0d0d0")
    draw.text((margin_x, 24), "Bandgap GNN 最终汇报", fill="#111111", font=meta_font)
    draw.text((margin_x, 62), f"{index}/{total}  {title}", fill="#111111", font=title_font)
    draw.text((img_w - 220, 28), "演讲稿", fill="#222222", font=meta_font)

    body = clean_latex(note)
    lines = wrap_text(draw, body, body_font, img_w - 2 * margin_x)
    y = 160
    line_h = 54
    for line in lines:
        if y + line_h > img_h - 55:
            draw.text((margin_x, y), "...", fill="#111111", font=body_font)
            break
        draw.text((margin_x, y), line, fill="#111111", font=body_font)
        y += line_h if line else line_h // 2

    image.save(path, "PDF", resolution=144.0)


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    notes = extract_braced_commands(tex, "speakernote")
    # Drop the macro definition itself if present.
    notes = [n for n in notes if not n.startswith("#1")]
    titles = extract_frame_titles(tex)

    slides = PdfReader(str(SLIDES_PDF))
    if len(notes) != len(slides.pages):
        raise RuntimeError(f"notes/pages mismatch: {len(notes)} notes vs {len(slides.pages)} pages")
    if len(titles) != len(slides.pages):
        raise RuntimeError(f"titles/pages mismatch: {len(titles)} titles vs {len(slides.pages)} pages")

    writer = PdfWriter()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        total = len(slides.pages)
        for idx, slide in enumerate(slides.pages, start=1):
            slide_page = copy(slide)
            slide_w = float(slide_page.mediabox.width)
            slide_h = float(slide_page.mediabox.height)

            note_pdf_path = tmp_path / f"note_{idx:03d}.pdf"
            make_note_pdf(notes[idx - 1], titles[idx - 1], idx, total, note_pdf_path)
            note_page = PdfReader(str(note_pdf_path)).pages[0]
            note_w = float(note_page.mediabox.width)
            note_h = float(note_page.mediabox.height)

            combined = PageObject.create_blank_page(width=slide_w * 2, height=slide_h)
            combined.merge_page(slide_page)
            transform = (
                Transformation()
                .scale(sx=slide_w / note_w, sy=slide_h / note_h)
                .translate(tx=slide_w, ty=0)
            )
            combined.merge_transformed_page(note_page, transform)
            writer.add_page(combined)

    with OUT_PDF.open("wb") as fh:
        writer.write(fh)
    print(f"Wrote {OUT_PDF} with {len(slides.pages)} side-by-side pages")


if __name__ == "__main__":
    main()
