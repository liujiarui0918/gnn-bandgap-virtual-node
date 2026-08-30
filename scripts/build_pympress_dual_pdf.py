#!/usr/bin/env python3
"""Build a Pympress side-by-side PDF without trusting Beamer's copied slide box.

The left half is imported from the verified normal slide PDF. The right half is
cropped from Beamer's native notes-on-second-screen PDF. This works around a
Beamer/pgfpages copy issue that can drop table/array content from selected left
slides while leaving the note-page thumbnail intact.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pymupdf


def build(normal_path: Path, native_notes_path: Path, output_path: Path) -> None:
    normal = pymupdf.open(normal_path)
    native = pymupdf.open(native_notes_path)
    if normal.page_count != native.page_count:
        raise ValueError(
            f"page count mismatch: normal={normal.page_count}, native={native.page_count}"
        )

    output = pymupdf.open()
    for index in range(normal.page_count):
        slide = normal[index]
        notes = native[index]
        slide_rect = slide.rect
        notes_rect = notes.rect
        if abs(notes_rect.height - slide_rect.height) > 0.1:
            raise ValueError(f"page {index + 1}: height mismatch")
        if notes_rect.width < 1.9 * slide_rect.width:
            raise ValueError(f"page {index + 1}: native notes PDF is not side-by-side")

        width, height = slide_rect.width, slide_rect.height
        page = output.new_page(width=2 * width, height=height)
        page.show_pdf_page(pymupdf.Rect(0, 0, width, height), normal, index)
        page.show_pdf_page(
            pymupdf.Rect(width, 0, 2 * width, height),
            native,
            index,
            clip=pymupdf.Rect(notes_rect.width / 2, 0, notes_rect.width, notes_rect.height),
        )

    metadata = normal.metadata.copy()
    title = metadata.get("title") or normal_path.stem
    metadata["title"] = f"{title} - Pympress notes right"
    metadata["subject"] = "Left: verified slide; right: speaker notes"
    output.set_metadata(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    output.save(output_path, garbage=4, deflate=True)
    output.close()
    native.close()
    normal.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("normal", type=Path)
    parser.add_argument("native_notes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.normal, args.native_notes, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
