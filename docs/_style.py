"""Check user-facing prose for the words the writing rules ban.

``python docs/_style.py`` lists every hit in the guides. ``conf.py`` runs the
same check on each page during the build, so a banned word fails ``-W``.
Code blocks, inline code and link targets are not checked.
"""
from pathlib import Path
import re
import sys

BANNED = (
    "owns", "owned", "owner", "owners", "ownership",
    "certify", "certifies", "certified", "certification",
    "eligibility", "eligible",
)
WORD = re.compile(r"\b(?:the|a|an|its|this|that) contracts?\b|\b(" + "|".join(BANNED) + r")\b", re.IGNORECASE)
FENCE = re.compile(r"^\s*(`{3,}|:{3,}|~{3,})")
INLINE = re.compile(r"`[^`]*`|\]\([^)]*\)|\(\S+\)=|\{[a-z]+\}`[^`]*`")
SKIP = ("maintainers", "development", "examples", "_build", "theme", "source")
GENERATED = ("reference/python", "reference/cpp")


def hits(text):
    """Yield (line number, word) for each banned word in prose."""
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        m = FENCE.match(line)
        if m and fence is None and m.group(1)[0] == "`":
            fence = m.group(1)
            continue
        if fence is not None:
            if line.strip().startswith(fence):
                fence = None
            continue
        for found in WORD.finditer(INLINE.sub("", line)):
            yield number, found.group(0).lower()


def checked(docname):
    """Whether a source document is user-facing prose."""
    return not (docname.split("/")[0] in SKIP or docname.startswith(GENERATED))


def main(root):
    total = 0
    for path in sorted(root.rglob("*.md")):
        if not checked(path.relative_to(root).with_suffix("").as_posix()):
            continue
        for number, word in hits(path.read_text()):
            print(f"{path.relative_to(root)}:{number}: {word}")
            total += 1
    print(f"{total} banned words")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(Path(__file__).resolve().parent))
