#!/usr/bin/env python3

import re
import sys
from pathlib import Path

MIN_LINES = 200
MIN_SECTIONS = 6
TOC_START = "<!-- BEGIN GENERATED TOC -->"
TOC_END = "<!-- END GENERATED TOC -->"

HEADING_PATTERN = re.compile(r"^(#{2,4})\s+(.+?)\s*#*\s*$")
LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
TAG_PATTERN = re.compile(r"<[^>]+>")


def _without_toc(text: str) -> str:
    pattern = re.compile(
        rf"\n*{re.escape(TOC_START)}.*?{re.escape(TOC_END)}\n*",
        re.DOTALL,
    )
    return pattern.sub("\n\n", text)


def _headings(text: str) -> list[tuple[int, str]]:
    headings = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING_PATTERN.match(line)
        if match:
            headings.append((len(match.group(1)), match.group(2)))
    return headings


def _anchor(title: str, counts: dict[str, int]) -> str:
    title = LINK_PATTERN.sub(r"\1", title)
    title = TAG_PATTERN.sub("", title)
    title = title.replace("`", "").lower()
    slug = re.sub(r"[^\w\- ]", "", title)
    slug = slug.replace(" ", "-")

    duplicate_index = counts.get(slug, 0)
    counts[slug] = duplicate_index + 1
    if duplicate_index:
        return f"{slug}-{duplicate_index}"
    return slug


def _render_toc(headings: list[tuple[int, str]]) -> str:
    counts: dict[str, int] = {}
    items = []
    for level, title in headings:
        indent = "  " * (level - 2)
        items.append(f"{indent}- [{title}](#{_anchor(title, counts)})")
    return "\n".join(
        [
            TOC_START,
            "## Contents",
            "",
            *items,
            TOC_END,
        ]
    )


def _insert_after_title(text: str, toc: str) -> str:
    lines = text.splitlines()
    title_index = next(
        index for index, line in enumerate(lines) if line.startswith("# ")
    )
    lines[title_index + 1 : title_index + 1] = ["", toc]
    return "\n".join(lines).rstrip() + "\n"


def update(path: Path) -> None:
    text = path.read_text()
    text_without_toc = _without_toc(text)
    headings = _headings(text_without_toc)
    needs_toc = (
        len(text_without_toc.splitlines()) >= MIN_LINES
        or len(headings) >= MIN_SECTIONS
    )

    updated = text_without_toc
    if needs_toc:
        updated = _insert_after_title(text_without_toc, _render_toc(headings))
    if updated != text:
        path.write_text(updated)


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        update(Path(filename))
