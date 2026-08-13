#!/usr/bin/env python3
"""Derive the Artifact-hosted build of the course from the standalone page.

Two builds of one page, from one source. The standalone file is a complete HTML
document and is what you host or open directly. The Artifact host supplies its
own doctype, head and body, and stamps `data-theme="light"|"dark"` on the root
element, so its build must be page CONTENT only and must not carry a theme
toggle of its own.

Three transforms, each for a reason that bit once:

* HTML comments are stripped **first**. The standalone file's integration note
  quotes the very tags this script then searches for, and extracting before
  stripping matches inside the comment.
* The page-level day/night toggle is removed. It writes `data-theme` on the root
  element, which is the host's job in an Artifact; leaving both in place means
  the page fights its container.
* `[data-theme="night"]` also answers to `:root[data-theme="dark"]`, which is the
  value the host actually stamps.

    python experiments/xp06_pruning/course/build_artifact.py [out.html]
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build() -> str:
    src = re.sub(r"<!--.*?-->", "", (HERE / "pruning-course.html").read_text(), flags=re.S)
    js = (HERE / "pruning-course.js").read_text()

    body = src.split("</head>", 1)[1]
    for tag in ("<body>", "</body>", "</html>"):
        body = body.replace(tag, "")
    body = re.sub(r'<script id="standalone-theme">.*?</script>', "", body, flags=re.S)
    body = re.sub(r'\s*<a id="daynight"[^>]*>.*?</a><span class="sep">/</span>', "",
                  body, flags=re.S)
    body = body.replace('<a class="mainTitle" href="/">※ pruning</a>',
                        '<span class="mainTitle">※ pruning</span>')
    body = re.sub(r'<script src="pruning-course\.js"></script>',
                  f"<script>\n{js}\n</script>", body)

    night = ':root[data-theme="dark"], [data-theme="night"]'
    styles = "\n".join(
        re.search(rf'<style id="{sid}">.*?</style>', src, re.S).group(0)
          .replace('[data-theme="night"]', night)
        for sid in ("kernwerk-base", "course-components"))

    out = f"<title>Pruning — build → shrink → measure</title>\n\n{styles}\n{body}"
    leaked = [t for t in ("<!DOCTYPE", "<html", "<head>", "<body>", "daynight()") if t in out]
    if leaked:
        raise SystemExit(f"wrapper leaked into the artifact build: {leaked}")
    return out


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "_artifact_build.html"
    dest.write_text(build())
    print(f"{dest} ({dest.stat().st_size} bytes)")
