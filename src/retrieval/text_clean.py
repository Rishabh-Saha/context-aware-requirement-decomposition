"""Strip Jira wiki-markup noise before chunking/embedding (CLAUDE.md retrieval constraints).

Jira descriptions and comments in this dump embed raw source dumps and stack traces inside
{code}/{noformat} blocks, plus bare URLs. None of that is prose a retrieval embedding should
represent, so it is removed rather than reformatted.

Scope, as the pipeline actually runs it: this cleanup is applied to the indexed past-ticket text
only (src/retrieval/sources.py). The retrieval query is the requirement's title and description as
recorded, uncleaned (src/pipeline/generate.py), so a query can carry formatting the corpus does
not. That asymmetry applies identically to every condition and is described in the thesis, so do
not "fix" it by cleaning the query here: doing so would change retrieval behaviour after the fact.
Note also what is not removed. Only {code}/{noformat} blocks, Java stack-trace lines, and bare URLs
go; other Jira wiki markup (headings, bold, list markers, tables) stays in the indexed text.
"""

from __future__ import annotations

import re

# {code} blocks may carry a language/title, e.g. {code:java} or {code:title=Foo.java}.
_CODE_BLOCK = re.compile(r"\{code(?::[^}]*)?\}.*?\{code\}", re.DOTALL | re.IGNORECASE)
_NOFORMAT_BLOCK = re.compile(r"\{noformat(?::[^}]*)?\}.*?\{noformat\}", re.DOTALL | re.IGNORECASE)
_URL = re.compile(r"https?://\S+")
_STACK_FRAME = re.compile(r"^[ \t]*(at\s+\S+\(.*\)|Caused by:.*|\.\.\.\s*\d+\s*more)\s*$", re.MULTILINE)
_BLANK_RUNS = re.compile(r"\n{3,}")


def strip_jira_markup(text: str | None) -> str:
    """Remove {code}/{noformat} blocks, raw URLs, and Java-style stack-trace lines; collapse the
    blank-line runs left behind. Returns "" for falsy input."""
    if not text:
        return ""
    cleaned = _CODE_BLOCK.sub("", text)
    cleaned = _NOFORMAT_BLOCK.sub("", cleaned)
    cleaned = _STACK_FRAME.sub("", cleaned)
    cleaned = _URL.sub("", cleaned)
    cleaned = _BLANK_RUNS.sub("\n\n", cleaned)
    return cleaned.strip()
