from src.ingestion.chunker import chunk_page

count = lambda s: len(s.split())   # simple stand-in tokenizer for tests


def test_breadcrumb_and_sections():
    body = "Intro text here for the page and more words to pass the minimum size.\n\n## Lifecycle\n\nA pod goes through phases and a lot of other words to make this body long enough.\n\n### Phases\n\nPending Running Succeeded Failed and some extra words to be above the minimum."
    chunks = chunk_page("Pods", body, count, 100, 10, min_body_tokens=5)
    assert [c.section for c in chunks] == ["Pods", "Pods > Lifecycle", "Pods > Lifecycle > Phases"]
    assert all(c.text.startswith(c.section) for c in chunks)


def test_hash_inside_code_is_not_a_heading():
    body = "## Setup\n\nRun this command now and then check the output carefully please.\n\n```bash\n# not a heading\nkubectl get pods\n```"
    chunks = chunk_page("T", body, count, 100, 10, min_body_tokens=5)
    assert len(chunks) == 1 and "# not a heading" in chunks[0].body


def test_chunks_respect_limit():
    para = " ".join(f"w{i}" for i in range(30))
    body = "## S\n\n" + "\n\n".join(para for _ in range(20))
    chunks = chunk_page("T", body, count, 100, 0, min_body_tokens=5)
    assert len(chunks) > 3
    assert all(count(c.text) <= 100 for c in chunks)


def test_code_block_kept_whole_when_it_fits():
    code = "```yaml\n" + "\n".join(f"key{i}: v" for i in range(20)) + "\n```"
    body = "## S\n\nIntro words here to keep things going for a while longer.\n\n" + code
    chunks = chunk_page("T", body, count, 100, 0, min_body_tokens=5)
    assert any(code in c.body for c in chunks)


def test_oversized_block_is_split():
    body = "## S\n\n" + "\n".join(f"line{i} " + "x " * 10 for i in range(40))
    chunks = chunk_page("T", body, count, 80, 0, min_body_tokens=5)
    assert len(chunks) > 1 and all(count(c.text) <= 80 for c in chunks)


def test_tiny_sections_skipped():
    chunks = chunk_page("T", "## Empty\n\nok", count, 100, 10, min_body_tokens=5)
    assert chunks == []
