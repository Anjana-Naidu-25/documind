"""Split a cleaned page into chunks.

Strategy
1. Split by Markdown headings, so a section stays together.
2. Inside a section, split into blocks (paragraphs, lists, code fences, tables).
   Code fences are never split unless one block alone exceeds the limit.
3. Pack blocks into chunks of at most `max_tokens`, repeating the last block
   of the previous chunk as overlap when it is small enough.
4. Prefix every chunk with a breadcrumb "Page > Section > Subsection" so a
   chunk that says "it restarts" still tells us what "it" is.
"""
import re
from dataclasses import dataclass
from typing import Callable

# Sections that are just link lists make poor answer material.
SKIP_SECTIONS = {"what's next", "whats next"}

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Chunk:
    text: str        # breadcrumb + body: this is what gets embedded
    section: str     # breadcrumb
    body: str
    chunk_idx: int


def _split_sections(body: str, title: str) -> list[tuple[list[str], str]]:
    """Return [(heading_path, section_text)], ignoring '#' lines inside code fences."""
    sections: list[tuple[list[str], str]] = []
    path: list[str] = []
    buf: list[str] = []
    in_fence = False

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append((list(path), text))
        buf.clear()

    for line in body.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            if level == 1:
                path[:] = []          # the page title is already in the breadcrumb
            else:
                path[:] = path[: level - 2] + [m.group(2)]
        else:
            buf.append(line)
    flush()
    return sections


def _split_blocks(section_text: str) -> list[str]:
    """Blank-line separated blocks, keeping fenced code together."""
    blocks, cur, in_fence = [], [], False
    for line in section_text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _split_oversized(block: str, limit: int, count: Callable[[str], int]) -> list[str]:
    """Break one block that is too big, line by line (then word by word if a line is huge)."""
    pieces, cur = [], []
    for line in block.splitlines():
        if count(line) > limit:                       # a single enormous line
            if cur:
                pieces.append("\n".join(cur)); cur = []
            words, w_cur = line.split(" "), []
            for w in words:
                if w_cur and count(" ".join(w_cur + [w])) > limit:
                    pieces.append(" ".join(w_cur)); w_cur = []
                w_cur.append(w)
            if w_cur:
                pieces.append(" ".join(w_cur))
            continue
        if cur and count("\n".join(cur + [line])) > limit:
            pieces.append("\n".join(cur)); cur = []
        cur.append(line)
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def chunk_page(
    title: str,
    body: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 350,
    overlap_tokens: int = 50,
    min_body_tokens: int = 12,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, section_text in _split_sections(body, title):
        # Skip "What's next" and everything nested under it (e.g. "What's next > Reference").
        if any(seg.replace("\u2019", "'").strip().lower() in SKIP_SECTIONS for seg in path):
            continue
        breadcrumb = " > ".join([title] + path) if title else " > ".join(path)
        prefix_cost = count_tokens(breadcrumb) + 2
        limit = max(max_tokens - prefix_cost, 50)

        blocks: list[str] = []
        for b in _split_blocks(section_text):
            blocks.extend([b] if count_tokens(b) <= limit else _split_oversized(b, limit, count_tokens))

        cur: list[str] = []
        cur_tokens = 0
        for b in blocks:
            bt = count_tokens(b)
            if cur and cur_tokens + bt > limit:
                _emit(chunks, breadcrumb, cur, count_tokens, min_body_tokens)
                last = cur[-1]
                if overlap_tokens and count_tokens(last) <= overlap_tokens and count_tokens(last) + bt <= limit:
                    cur, cur_tokens = [last], count_tokens(last)
                else:
                    cur, cur_tokens = [], 0
            cur.append(b)
            cur_tokens += bt
        if cur:
            _emit(chunks, breadcrumb, cur, count_tokens, min_body_tokens)
    return chunks


def _emit(chunks, breadcrumb, blocks, count_tokens, min_body_tokens) -> None:
    body = "\n\n".join(blocks).strip()
    if not body or count_tokens(body) < min_body_tokens:   # too small to be useful on its own
        return
    chunks.append(Chunk(text=f"{breadcrumb}\n\n{body}", section=breadcrumb, body=body, chunk_idx=len(chunks)))
