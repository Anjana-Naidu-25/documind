"""Find the Markdown pages and work out each page's public URL."""
from dataclasses import dataclass
from pathlib import Path

from src.config import RAW_DIR, SITE_BASE_URL


@dataclass
class RawDoc:
    rel_path: str   # e.g. concepts/workloads/pods/_index.md
    url: str        # e.g. https://kubernetes.io/docs/concepts/workloads/pods/
    text: str


def path_to_url(rel_path: str, base_url: str = SITE_BASE_URL) -> str:
    p = Path(rel_path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "_index":
        parts = parts[:-1]
    return base_url + "/".join(parts) + "/"


def load_docs(raw_dir: Path = RAW_DIR) -> list[RawDoc]:
    docs = []
    for f in sorted(raw_dir.rglob("*.md")):
        rel = f.relative_to(raw_dir).as_posix()
        text = f.read_text(encoding="utf-8", errors="ignore")
        docs.append(RawDoc(rel_path=rel, url=path_to_url(rel), text=text))
    return docs
