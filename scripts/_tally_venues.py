"""Tally the journal/conference venues appearing in the paper bibliography."""

import re
import collections
import pathlib

TEX = (
    pathlib.Path(__file__).resolve().parents[1]
    / "gnn_bandgap_research_paper_draft_en.tex"
)

text = TEX.read_text(encoding="utf-8")
bib = text.split(r"\begin{thebibliography}")[1]
entries = bib.split(r"\bibitem")[1:]

counter = collections.Counter()
for entry in entries:
    venues = re.findall(r"\\emph\{([^}]+)\}", entry)
    if venues:
        counter[venues[0].strip()] += 1

for venue, n in counter.most_common():
    print(f"{n:2d} | {venue}")
