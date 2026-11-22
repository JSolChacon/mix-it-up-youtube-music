"""Renders the key code fragments to PNG images for the documentation.

Usage:  python tools/render_code_images.py
Each snippet is cut from the real source file, so images never drift from the code.
"""
from pathlib import Path

from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"

# (image name, file, first line that starts the snippet, line that ends it - exclusive)
SNIPPETS = [
    ("code_01_features.png", "src/data.py", "def build_features", None),
    ("code_02_baseline.png", "src/recommender.py", "def taste_vector", "# 2. Mix It Up"),
    ("code_03_profile.png", "src/recommender.py", "    # -- step 2: robust taste profile", "    # -- steps 3 + 4"),
    ("code_04_recommend.png", "src/recommender.py", "    # -- steps 3 + 4", "        out = df.loc[picks]"),
    ("code_05_mmr.png", "src/recommender.py", "    def _mmr", "    # -- step 5"),
    ("code_06_explain.png", "src/recommender.py", "    # -- step 5", None),
    ("code_07_users.png", "src/evaluate.py", "def make_history", "def metrics"),
]


def cut(text, start, end):
    i = text.index(start)
    j = text.index(end, i + 1) if end else len(text)
    lines = text[i:j].rstrip().splitlines()
    while lines and lines[-1].lstrip().startswith("#"):   # drop trailing section dividers
        lines.pop()
    return "\n".join(lines).rstrip() + "\n"


for name, file, start, end in SNIPPETS:
    src = (ROOT / file).read_text(encoding="utf-8")
    snippet = cut(src, start, end)
    first_line = src[: src.index(start)].count("\n") + 1
    fmt = ImageFormatter(style="one-dark", font_name="Consolas", font_size=22,
                         line_numbers=True, line_number_start=first_line,
                         line_number_bg="#21252b", line_number_fg="#636d83",
                         line_pad=6, image_pad=24)
    (OUT / name).write_bytes(highlight(snippet, PythonLexer(), fmt))
    print("saved", name, f"({file} from line {first_line})")
