"""Turn fetched bytes into ordered sections, keeping headings and PDF page numbers."""

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf
import trafilatura
from bs4 import BeautifulSoup
from lxml import etree

from researchos.fetch import Fetched
from researchos.models import Section

MAX_CHARS = 500_000
MAX_PAGES = 100
MIN_CHARS = 200
HEADINGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
BLOCKS = [*HEADINGS, "p", "li", "pre", "blockquote", "dd", "dt", "figcaption", "tr"]
BOILERPLATE = [
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "nav",
    "aside",
    "footer",
    "form",
    "iframe",
    "button",
    "select",
]
ARXIV_STAMP = re.compile(r"^arXiv:\S+\s+\[")


class ParseFailure(Exception):
    pass


@dataclass
class Parsed:
    title: str
    sections: list[Section]
    pages: int | None = None
    truncated: bool = False


def clean(text: str) -> str:
    return " ".join(text.split())


@dataclass
class SectionBuilder:
    sections: list[Section] = field(default_factory=list)
    heading: str = ""
    page: int | None = None
    paragraphs: list[str] = field(default_factory=list)
    chars: int = 0
    truncated: bool = False

    def add(self, paragraph: str) -> None:
        if self.truncated or not paragraph:
            return
        if self.chars + len(paragraph) > MAX_CHARS:
            self.truncated = True
            return
        self.chars += len(paragraph)
        self.paragraphs.append(paragraph)

    def flush(self) -> None:
        if self.paragraphs:
            self.sections.append(
                Section(heading=self.heading, text="\n\n".join(self.paragraphs), page=self.page)
            )
        self.paragraphs = []


def finish(title: str, builder: SectionBuilder, pages: int | None, truncated: bool) -> Parsed:
    builder.flush()
    if builder.chars < MIN_CHARS:
        raise ParseFailure(
            "no readable text found (the page may need JavaScript, or the PDF may be scanned)"
        )
    return Parsed(title, builder.sections, pages, truncated or builder.truncated)


def pdf_link(body: bytes) -> str | None:
    """The full-text PDF a scholarly landing page advertises, if any."""
    soup = BeautifulSoup(body, "html.parser")
    tag = soup.find("meta", attrs={"name": "citation_pdf_url"})
    return (tag.get("content") or None) if tag else None


def parse_article(body: bytes) -> Parsed:
    """Main-content extraction: drops navigation, sidebars, and other page chrome."""
    xml = trafilatura.extract(
        body,
        output_format="xml",
        with_metadata=True,
        include_tables=True,
        include_links=False,
        include_images=False,
        include_comments=False,
        favor_recall=True,
    )
    if not xml:
        raise ParseFailure("no article content found")
    root = etree.fromstring(xml.encode())
    builder = SectionBuilder()
    blocks = ("p", "item", "quote", "code", "row")
    for node in root.iter("head", *blocks):
        if next(node.iterancestors("row"), None) is not None:
            continue
        if node.tag == "row":
            cells = [clean("".join(cell.itertext())) for cell in node.findall("cell")]
            text = " | ".join(cells) if any(cells) else ""
        elif next(node.iterdescendants(*blocks), None) is not None:
            continue
        elif node.tag == "code":
            text = "".join(node.itertext()).strip()
        else:
            text = clean("".join(node.itertext()))
        if not text:
            continue
        if node.tag == "head":
            builder.flush()
            builder.heading = text
        else:
            builder.add(text)
    return finish(clean(root.get("title") or ""), builder, None, False)


def parse_html(body: bytes, charset: str | None = None) -> Parsed:
    try:
        return parse_article(body)
    except ParseFailure:
        return parse_blocks(body, charset)


def parse_blocks(body: bytes, charset: str | None = None) -> Parsed:
    """Fallback for pages the article extractor rejects: every text block under the body."""
    soup = BeautifulSoup(body, "html.parser", from_encoding=charset)
    title = clean(soup.title.get_text()) if soup.title else ""
    for tag in soup(BOILERPLATE):
        tag.decompose()
    for tag in soup("br"):
        tag.replace_with("\n")
    candidates = [node for node in (soup.find("main"), soup.find("article")) if node is not None]
    candidates.append(soup.body or soup)
    root = next(
        (node for node in candidates if len(clean(node.get_text())) >= MIN_CHARS), candidates[-1]
    )
    builder = SectionBuilder()
    for tag in root.find_all(BLOCKS):
        if tag.find_parent("tr"):
            continue
        if tag.name == "tr":
            cells = [clean(cell.get_text()) for cell in tag.find_all(["th", "td"], recursive=False)]
            text = " | ".join(cells) if any(cells) else ""
        elif tag.find(BLOCKS):
            continue
        elif tag.name == "pre":
            text = tag.get_text().strip()
        else:
            text = clean(tag.get_text())
        if not text:
            continue
        if tag.name in HEADINGS:
            builder.flush()
            builder.heading = text
        else:
            builder.add(text)
    full = [clean(line) for line in root.get_text("\n").splitlines()]
    full = [line for line in full if line]
    # Pages that put prose directly in <div>s defeat block extraction; keep their text whole.
    if builder.chars < 0.5 * sum(map(len, full)):
        builder = SectionBuilder()
        for line in full:
            builder.add(line)
    return finish(title, builder, None, False)


def parse_text(body: bytes, charset: str | None = None) -> Parsed:
    builder = SectionBuilder()
    for paragraph in re.split(r"\n\s*\n", body.decode(charset or "utf-8", errors="replace")):
        builder.add(clean(paragraph))
    return finish("", builder, None, False)


def pdf_lines(pdf) -> list[tuple[int, int, str, float]]:
    lines = []
    for number in range(min(pdf.page_count, MAX_PAGES)):
        for index, block in enumerate(pdf[number].get_text("dict")["blocks"]):
            for line in block.get("lines", []):
                # Spaces are often separate spans, so join everything before cleaning.
                text = clean("".join(span["text"] for span in line["spans"]))
                if text and not ARXIV_STAMP.match(text):
                    size = max(line["spans"], key=lambda span: len(span["text"].strip()))["size"]
                    lines.append((number + 1, index, text, round(size, 1)))
    return lines


def parse_pdf(body: bytes) -> Parsed:
    try:
        pdf = pymupdf.open(stream=body, filetype="pdf")
    except Exception:
        raise ParseFailure("PDF could not be opened") from None
    with pdf:
        if pdf.needs_pass:
            raise ParseFailure("PDF is password-protected")
        pages = pdf.page_count
        title = clean((pdf.metadata or {}).get("title") or "")
        try:
            lines = pdf_lines(pdf)
        except Exception:
            raise ParseFailure("PDF text could not be extracted") from None
    if not lines:
        raise ParseFailure("no readable text found (the PDF may be scanned images)")
    sizes = Counter()
    for _, _, text, size in lines:
        sizes[size] += len(text)
    body_size = sizes.most_common(1)[0][0]
    builder = SectionBuilder()
    paragraph: list[str] = []
    last_block = None
    heading_open = False

    def end_paragraph() -> None:
        builder.add(" ".join(paragraph))
        paragraph.clear()

    for page, block, text, size in lines:
        # Headings are detected by font size: clearly larger than body text, and short.
        if size >= body_size + 0.9 and len(text) <= 150 and any(c.isalpha() for c in text):
            end_paragraph()
            builder.flush()
            builder.heading = f"{builder.heading} {text}" if heading_open else text
            builder.page, heading_open = page, True
            continue
        heading_open = False
        if page != builder.page:
            # One section never spans pages, so every passage keeps an exact page number.
            end_paragraph()
            builder.flush()
            builder.page = page
        if (page, block) != last_block:
            end_paragraph()
            last_block = (page, block)
        if paragraph and paragraph[-1].endswith("-") and text[:1].islower():
            paragraph[-1] = paragraph[-1][:-1] + text
        else:
            paragraph.append(text)
    end_paragraph()
    return finish(title, builder, pages, pages > MAX_PAGES)


def parse(fetched: Fetched) -> Parsed:
    try:
        if fetched.kind == "pdf":
            return parse_pdf(fetched.body)
        if fetched.kind == "text":
            return parse_text(fetched.body, fetched.charset)
        return parse_html(fetched.body, fetched.charset)
    except ParseFailure:
        raise
    except Exception:
        raise ParseFailure("unexpected parser failure") from None
