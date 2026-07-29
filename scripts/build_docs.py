#!/usr/bin/env python3
"""Build the dependency-free static documentation site."""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "html"
INDEX_PATH = OUTPUT_DIR / "index.html"
DOC_LINK_START = "<!-- GENERATED-DOC-LINKS:START -->"
DOC_LINK_END = "<!-- GENERATED-DOC-LINKS:END -->"

FENCE_RE = re.compile(r"^```(?P<language>[\w+-]*)\s*$")
HEADING_RE = re.compile(r"^(?P<level>#{1,6})\s+(?P<text>.+?)\s*$")
LIST_RE = re.compile(r"^\s*(?P<marker>[-+*]|\d+\.)\s+(?P<text>.+)$")
TABLE_DIVIDER_RE = re.compile(r"^:?-{3,}:?$")


def render_inline(value: str) -> str:
    """Render the small inline Markdown subset used by project documents."""

    rendered = html.escape(value, quote=True)
    placeholders: list[str] = []

    def stash(fragment: str) -> str:
        token = f"@@NEG_CYCLE_INLINE_{len(placeholders)}@@"
        placeholders.append(fragment)
        return token

    rendered = re.sub(
        r"`([^`\n]+)`",
        lambda match: stash(f"<code>{match.group(1)}</code>"),
        rendered,
    )
    rendered = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        lambda match: (
            f'<a href="{match.group(2)}">{match.group(1)}</a>'
        ),
        rendered,
    )
    rendered = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", rendered)

    for index, fragment in enumerate(placeholders):
        rendered = rendered.replace(f"@@NEG_CYCLE_INLINE_{index}@@", fragment)
    return rendered


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_table_divider(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(TABLE_DIVIDER_RE.fullmatch(cell) for cell in cells)


def is_block_start(lines: list[str], index: int) -> bool:
    line = lines[index]
    if FENCE_RE.match(line) or HEADING_RE.match(line) or line.startswith(">"):
        return True
    if LIST_RE.match(line):
        return True
    return (
        index + 1 < len(lines)
        and "|" in line
        and is_table_divider(lines[index + 1])
    )


def render_markdown(source: str, title: str) -> str:
    """Render project Markdown without requiring a third-party package."""

    lines = source.splitlines()
    rendered: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue

        fence = FENCE_RE.match(line)
        if fence:
            language = fence.group("language")
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            language_class = (
                f' class="language-{html.escape(language, quote=True)}"'
                if language
                else ""
            )
            code = html.escape("\n".join(code_lines), quote=False)
            rendered.append(
                f'<div class="code-block"><pre><code{language_class}>'
                f"{code}</code></pre></div>"
            )
            continue

        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group("level"))
            if level == 1 and heading.group("text").strip() == title:
                index += 1
                continue
            rendered.append(
                f"<h{level}>{render_inline(heading.group('text'))}</h{level}>"
            )
            index += 1
            continue

        if (
            index + 1 < len(lines)
            and "|" in line
            and is_table_divider(lines[index + 1])
        ):
            headers = split_table_row(line)
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(split_table_row(lines[index]))
                index += 1

            header_html = "".join(
                f'<th scope="col">{render_inline(cell)}</th>' for cell in headers
            )
            body_rows = []
            for row in rows:
                padded = row + [""] * (len(headers) - len(row))
                cells = "".join(
                    f"<td>{render_inline(cell)}</td>" for cell in padded[: len(headers)]
                )
                body_rows.append(f"<tr>{cells}</tr>")
            rendered.append(
                '<div class="table-wrap"><table><thead><tr>'
                f"{header_html}</tr></thead><tbody>{''.join(body_rows)}"
                "</tbody></table></div>"
            )
            continue

        if line.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].startswith(">"):
                quote_lines.append(lines[index][1:].strip())
                index += 1
            rendered.append(
                f"<blockquote><p>{render_inline(' '.join(quote_lines))}</p></blockquote>"
            )
            continue

        list_match = LIST_RE.match(line)
        if list_match:
            ordered = list_match.group("marker")[0].isdigit()
            list_tag = "ol" if ordered else "ul"
            items: list[str] = []
            while index < len(lines):
                item_match = LIST_RE.match(lines[index])
                if not item_match:
                    break
                item_ordered = item_match.group("marker")[0].isdigit()
                if item_ordered != ordered:
                    break

                item_lines = [item_match.group("text").strip()]
                index += 1
                while index < len(lines):
                    continuation = lines[index]
                    if not continuation.strip() or LIST_RE.match(continuation):
                        break
                    if continuation.startswith((" ", "\t")):
                        item_lines.append(continuation.strip())
                        index += 1
                        continue
                    break
                items.append(render_inline(" ".join(item_lines)))

                next_index = index
                while next_index < len(lines) and not lines[next_index].strip():
                    next_index += 1
                next_item = (
                    LIST_RE.match(lines[next_index])
                    if next_index < len(lines)
                    else None
                )
                if next_item is not None:
                    next_ordered = next_item.group("marker")[0].isdigit()
                    if next_ordered == ordered:
                        index = next_index
                        continue
                index = next_index
                break

            item_html = "".join(f"<li>{item}</li>" for item in items)
            rendered.append(
                f'<{list_tag} class="feature-list">{item_html}</{list_tag}>'
            )
            continue

        paragraph_lines = [line.strip()]
        index += 1
        while index < len(lines) and lines[index].strip():
            if is_block_start(lines, index):
                break
            paragraph_lines.append(lines[index].strip())
            index += 1
        rendered.append(f"<p>{render_inline(' '.join(paragraph_lines))}</p>")

    return "\n".join(rendered)


def document_title(source: str, fallback: str) -> str:
    for line in source.splitlines():
        heading = re.match(r"^#\s+(.+?)\s*$", line)
        if heading:
            return heading.group(1)
    return fallback


def document_page(source_path: Path) -> tuple[str, str]:
    markdown = source_path.read_text(encoding="utf-8")
    title = document_title(markdown, source_path.stem.replace("_", " ").title())
    safe_title = html.escape(title, quote=True)
    safe_name = html.escape(source_path.name, quote=True)
    browser_title = title if title == "NegCycle" else f"{title} · NegCycle"
    safe_browser_title = html.escape(browser_title, quote=True)
    body = render_markdown(markdown, title)

    page = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{safe_browser_title}</title>
    <meta
      name="description"
      content="{safe_title}, compiled from {safe_name} for the NegCycle documentation site."
    >
    <link rel="stylesheet" href="./styles.css">
  </head>
  <body>
    <div class="page-shell">
      <header class="hero">
        <div class="hero-copy">
          <p class="eyebrow">Project Document</p>
          <h1>{safe_title}</h1>
          <p class="lede">
            Compiled from <code>{safe_name}</code> by the dependency-free
            documentation build.
          </p>
          <div class="hero-actions">
            <a class="button button-primary" href="./index.html">NegCycle home</a>
            <a class="button button-secondary" href="./index.html#documents">
              All documents
            </a>
          </div>
        </div>
        <div class="hero-card">
          <div class="terminal">
            <div class="terminal-bar">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <pre><code>{safe_name}
→ html/{html.escape(source_path.stem, quote=True)}.html</code></pre>
          </div>
        </div>
      </header>

      <main>
        <section class="grid-section">
          <article class="panel panel-wide document-content">
{body}
          </article>
        </section>
      </main>

      <footer class="site-footer">
        <p><a href="./index.html">NegCycle documentation</a></p>
        <p>
          Created by <a href="https://goblinreactor.com/">Goblin Reactor</a>.
        </p>
      </footer>
    </div>
  </body>
</html>
"""
    return title, page


def source_sort_key(path: Path) -> tuple[int, str]:
    return (0 if path.name.casefold() == "readme.md" else 1, path.name.casefold())


def update_document_links(documents: list[tuple[Path, str]]) -> None:
    index = INDEX_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^(?P<indent>[ \t]*){re.escape(DOC_LINK_START)}.*?"
        rf"^[ \t]*{re.escape(DOC_LINK_END)}",
        re.MULTILINE | re.DOTALL,
    )

    match = pattern.search(index)
    if match is None:
        raise RuntimeError(f"document link markers are missing from {INDEX_PATH}")

    indent = match.group("indent")
    links = [f"{indent}{DOC_LINK_START}"]
    for source_path, title in documents:
        target = f"{source_path.stem}.html"
        links.append(
            f'{indent}<li><a href="./{html.escape(target, quote=True)}">'
            f"{html.escape(title)}</a> — compiled from "
            f"<code>{html.escape(source_path.name)}</code></li>"
        )
    links.append(f"{indent}{DOC_LINK_END}")

    updated = pattern.sub("\n".join(links), index, count=1)
    INDEX_PATH.write_text(updated, encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(ROOT.glob("*.md"), key=source_sort_key)
    if not sources:
        raise RuntimeError("no top-level Markdown documents found")

    documents: list[tuple[Path, str]] = []
    rendered_pages: list[tuple[Path, str]] = []
    for source_path in sources:
        title, page = document_page(source_path)
        documents.append((source_path, title))
        rendered_pages.append((OUTPUT_DIR / f"{source_path.stem}.html", page))

    update_document_links(documents)
    for output_path, page in rendered_pages:
        output_path.write_text(page, encoding="utf-8")

    for benchmark_name in ("benchmarks.csv", "headline_benchmarks.csv"):
        benchmark_data = ROOT / benchmark_name
        if benchmark_data.exists():
            shutil.copyfile(benchmark_data, OUTPUT_DIR / benchmark_data.name)

    print(
        f"Built {len(rendered_pages)} Markdown page(s) in "
        f"{OUTPUT_DIR.relative_to(ROOT)}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
