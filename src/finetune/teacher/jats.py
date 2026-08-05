"""JATS XML helpers, copied from the corpus text-cache module.

Only the pieces the teacher pipeline needs: article-title / abstract / body
extraction and tag-stripping. Regex (fast, matches the corpus tooling's style).
NOTE: this module's `_WS` collapses horizontal whitespace only ([ \\t\\r\\f\\v]+);
the guard normaliser in `pipeline._norm` uses a different, full-\\s+ collapse.
"""
from __future__ import annotations

import re

_TITLE = re.compile(r"<article-title[^>]*>(.*?)</article-title>", re.S | re.I)
_ABS = re.compile(r"<abstract[^>]*>(.*?)</abstract>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
# body-only extraction: JATS wraps the real article in <body>; <front> is journal/
# author/copyright meta, <back> is references + ack/funding/ethics/consent/contribs.
_BODY = re.compile(r"<body\b[^>]*>(.*?)</body>", re.S | re.I)


def _clean(s: str) -> str:
    return _WS.sub(" ", _TAG.sub(" ", s)).strip()
