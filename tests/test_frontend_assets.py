"""Static frontend contracts: asset wiring and layout rules that broke before."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "static"
PAGES = sorted(STATIC.glob("*.html"))
CSS = "\n".join(path.read_text() for path in sorted((STATIC / "css").glob("*.css")))


def rule(selector: str) -> str:
    """Return the declaration block of the first exact selector match."""
    match = re.search(rf"(?:^|\n){re.escape(selector)}\s*\{{([^}}]*)\}}", CSS)
    assert match, f"missing CSS rule: {selector}"
    return match.group(1)


def test_pages_exist():
    names = {page.name for page in PAGES}
    assert names == {
        "index.html",
        "overview.html",
        "prepare.html",
        "explore.html",
        "model.html",
        "insights.html",
    }


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
def test_linked_assets_resolve(page):
    html = page.read_text()

    for href in re.findall(r'<link[^>]+href="(/static/[^"]+)"', html):
        assert (STATIC / href.removeprefix("/static/")).is_file(), href

    for src in re.findall(r'<script[^>]+src="(/static/[^"]+)"', html):
        assert (STATIC / src.removeprefix("/static/")).is_file(), src


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
def test_shared_stylesheets_are_loaded(page):
    html = page.read_text()
    for sheet in ("tokens.css", "layout.css", "components.css", "pages.css"):
        assert f"/static/css/{sheet}" in html, (page.name, sheet)


def test_model_option_box_is_a_block():
    """An inline wrapper around the flex header drew fragmented borders."""
    assert "display: block" in rule(".model-option__box")
    assert "display: flex" in rule(".model-option__top")


def test_charts_fill_their_host():
    assert "width: 100%" in rule(".chart-svg")
    assert "height: auto" in rule(".chart-svg")


def test_upload_accepts_every_supported_extension():
    html = (STATIC / "index.html").read_text()
    accept = re.search(r'id="file-input"[^>]*accept="([^"]+)"', html)
    assert accept, "file input needs an accept list"
    for extension in (".csv", ".tsv", ".txt", ".xlsx", ".xls"):
        assert extension in accept.group(1)


def test_uploads_are_not_written_to_disk():
    """The uploader must stay in-memory: no temp-file or open() writes."""
    for path in (Path(__file__).resolve().parent.parent / "src").glob("*.py"):
        source = path.read_text()
        assert "tempfile" not in source, path.name
