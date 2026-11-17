"""Turn a raw Kubernetes (Hugo) Markdown page into clean text.

Hugo pages contain front matter and shortcodes such as {{< note >}} or
{{< glossary_tooltip text="pods" term_id="pod" >}}. If we embedded those
as-is, the markup would pollute the vectors and the answers.
"""
import re

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# {{< name args >}}  or  {{% name args %}}   (also the closing {{< /name >}})
_OPEN = r"\{\{[<%]\s*"
_CLOSE = r"\s*[>%]\}\}"

ADMONITIONS = {"note": "Note", "caution": "Caution", "warning": "Warning"}

HEADING_NAMES = {
    "prerequisites": "Before you begin",
    "objectives": "Objectives",
    "lessoncontent": "Lesson content",
    "cleanup": "Cleaning up",
    "whatsnext": "What's next",
    "seealso": "See also",
}


def _attr(attrs: str, name: str) -> str | None:
    m = re.search(rf'{name}\s*=\s*"([^"]*)"', attrs)
    return m.group(1) if m else None


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Return (metadata, body). Only title/description are needed, so a tiny parser is enough."""
    m = FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta = {}
    for key in ("title", "description", "linkTitle"):
        km = re.search(rf'^{key}:\s*(.+?)\s*$', m.group(1), re.MULTILINE)
        if km:
            meta[key] = km.group(1).strip().strip("\"'")
    return meta, text[m.end():]


def _glossary_tooltip(m: re.Match) -> str:
    attrs = m.group(1)
    return _attr(attrs, "text") or (_attr(attrs, "term_id") or "").replace("-", " ")


def _glossary_definition(m: re.Match) -> str:
    return (_attr(m.group(1), "term_id") or "").replace("-", " ")


def _feature_state(m: re.Match) -> str:
    attrs = m.group(1)
    state = _attr(attrs, "state") or ""
    version = _attr(attrs, "for_k8s_version") or ""
    parts = [p for p in (f"Kubernetes {version}" if version else "", state) if p]
    return f"FEATURE STATE: {', '.join(parts)}" if parts else ""


def _heading(m: re.Match) -> str:
    key = m.group(1)
    return HEADING_NAMES.get(key.lower(), key)   # the "##" is already in the source line


def _figure(m: re.Match) -> str:
    return _attr(m.group(1), "caption") or ""


def clean_markdown(text: str) -> tuple[dict, str]:
    """Return (metadata, cleaned_body)."""
    meta, body = parse_front_matter(text)

    body = HTML_COMMENT.sub("", body)

    # Diagram source is useless as text: drop the whole block.
    body = re.sub(
        _OPEN + r"mermaid[^>%]*" + _CLOSE + r".*?" + _OPEN + r"/\s*mermaid" + _CLOSE,
        "", body, flags=re.DOTALL,
    )

    # {{< highlight yaml >}} ... {{< /highlight >}}  ->  fenced code block
    body = re.sub(_OPEN + r"highlight\s*([a-zA-Z0-9_+-]*)[^>%]*" + _CLOSE, r"```\1", body)
    body = re.sub(_OPEN + r"/\s*highlight" + _CLOSE, "```", body)

    # Notes, cautions, warnings -> "Note:" etc. (opening tag only; closing tag removed)
    for tag, label in ADMONITIONS.items():
        body = re.sub(_OPEN + tag + r"[^>%]*" + _CLOSE, f"**{label}:** ", body)
        body = re.sub(_OPEN + r"/\s*" + tag + _CLOSE, "", body)

    body = re.sub(_OPEN + r"glossary_tooltip\s+([^>%]*?)" + _CLOSE, _glossary_tooltip, body, flags=re.DOTALL)
    body = re.sub(_OPEN + r"glossary_definition\s+([^>%]*?)" + _CLOSE, _glossary_definition, body, flags=re.DOTALL)
    body = re.sub(_OPEN + r"feature-state\s+([^>%]*?)" + _CLOSE, _feature_state, body, flags=re.DOTALL)
    body = re.sub(_OPEN + r'heading\s+"([^"]+)"[^>%]*' + _CLOSE, _heading, body)
    body = re.sub(_OPEN + r"figure\s+([^>%]*?)" + _CLOSE, _figure, body, flags=re.DOTALL)

    # code_sample file="x/y.yaml" -> short reference (the manifest itself is not inlined)
    body = re.sub(
        _OPEN + r"code_sample\s+([^>%]*?)" + _CLOSE,
        lambda m: f"(Example manifest: {_attr(m.group(1), 'file') or ''})",
        body, flags=re.DOTALL,
    )

    # Tabs: keep the content, label each tab.
    body = re.sub(_OPEN + r'tab\s+[^>%]*?name\s*=\s*"([^"]+)"[^>%]*' + _CLOSE, r"\n**\1:**\n", body)
    body = re.sub(_OPEN + r"/?\s*tabs?[^>%]*" + _CLOSE, "", body)

    # Version placeholders: we can't know the release, so say it generically.
    body = re.sub(_OPEN + r"skew\s+[^>%]*" + _CLOSE, "the current version", body)

    # Anything else (include, param, version-check, thirdparty-content, table, api-reference, ...)
    body = re.sub(_OPEN + r"/?\s*[a-zA-Z_-]+[^>%]*?" + _CLOSE, "", body, flags=re.DOTALL)

    # Tidy whitespace.
    body = "\n".join(line.rstrip() for line in body.splitlines())
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if not body and meta.get("description"):
        body = meta["description"]          # index pages: keep their one-line summary
    return meta, body
