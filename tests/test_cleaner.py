from src.ingestion.cleaner import clean_markdown
from src.ingestion.loader import path_to_url


def test_front_matter_and_comments_removed():
    meta, body = clean_markdown("---\ntitle: Pods\nweight: 10\n---\n\n<!-- overview -->\nHello\n")
    assert meta["title"] == "Pods"
    assert body == "Hello"


def test_glossary_tooltip_keeps_text():
    _, body = clean_markdown('A group of {{< glossary_tooltip text="containers" term_id="container" >}}.')
    assert body == "A group of containers."


def test_note_becomes_label():
    _, body = clean_markdown("{{< note >}}\nBe careful.\n{{< /note >}}")
    assert "**Note:**" in body and "Be careful." in body and "{{" not in body


def test_closing_tag_with_space_removed():
    _, body = clean_markdown("{{< warning >}}\nx\n{{</ warning >}}")
    assert "{{" not in body


def test_heading_shortcode():
    _, body = clean_markdown('## {{% heading "whatsnext" %}}\n\nRead more.')
    assert body.startswith("## What's next")


def test_feature_state():
    _, body = clean_markdown('{{< feature-state for_k8s_version="v1.25" state="stable" >}}')
    assert "stable" in body and "v1.25" in body


def test_mermaid_block_dropped():
    _, body = clean_markdown("Before\n{{< mermaid >}}\ngraph TD; A-->B\n{{</ mermaid >}}\nAfter")
    assert "graph" not in body and "Before" in body and "After" in body


def test_unknown_shortcode_removed():
    _, body = clean_markdown("Text {{< some-new-thing a=\"b\" >}} more")
    assert "{{" not in body


def test_go_template_in_code_untouched():
    _, body = clean_markdown("kubectl get -o jsonpath='{{.items}}'")
    assert "{{.items}}" in body


def test_url_building():
    assert path_to_url("concepts/workloads/pods/_index.md") == "https://kubernetes.io/docs/concepts/workloads/pods/"
    assert path_to_url("concepts/configuration/secret.md") == "https://kubernetes.io/docs/concepts/configuration/secret/"


def test_heading_anchor_stripped():
    _, body = clean_markdown("## Container restarts {#restart-policy}\n\ntext")
    assert body.startswith("## Container restarts") and "{#" not in body


def test_glossary_definition_dropped():
    _, body = clean_markdown('Title\n\n{{< glossary_definition term_id="configmap" length="short" >}}\n\nMore')
    assert "configmap" not in body.lower()
