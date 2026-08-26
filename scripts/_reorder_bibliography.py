"""Reorder the manual thebibliography so that reference numbers follow the order
of first citation in the body, and sort the keys inside every multi-key \\cite
group so that printed number lists are ascending.

Sorting keys within a group is a fixed point of the numbering rule: new keys in
a group are numbered in written order, so sorting by assigned number leaves
their relative order unchanged and the numbering stays valid.
"""

import re
import shutil
import sys

PAPER = "gnn_bandgap_research_paper_draft_en.tex"
MARK = r"\begin{thebibliography}"

tex = open(PAPER, encoding="utf-8").read()
head, tail = tex.split(MARK, 1)
bib_body, after = tail.split(r"\end{thebibliography}", 1)

# 1. first-citation order over the body
order, seen = [], set()
for m in re.finditer(r"\\cite\{([^}]*)\}", head):
    for k in [k.strip() for k in m.group(1).split(",")]:
        if k not in seen:
            seen.add(k)
            order.append(k)

# 2. split the bibliography into per-entry blocks
starts = [m.start() for m in re.finditer(r"\\bibitem\{", bib_body)]
if not starts:
    sys.exit("no \\bibitem found")
preamble = bib_body[: starts[0]]
blocks = {}
for i, s in enumerate(starts):
    e = starts[i + 1] if i + 1 < len(starts) else len(bib_body)
    block = bib_body[s:e]
    key = re.match(r"\\bibitem\{([^}]*)\}", block).group(1)
    blocks[key] = block.rstrip() + "\n\n"

missing = [k for k in order if k not in blocks]
extra = [k for k in blocks if k not in seen]
if missing or extra:
    sys.exit(f"key mismatch: cited-but-undefined={missing} defined-but-uncited={extra}")

# 3. rebuild the bibliography in first-citation order
new_bib = preamble + "".join(blocks[k] for k in order)
number = {k: i + 1 for i, k in enumerate(order)}


# 4. sort keys inside each \cite group by the new number
def fix_group(m):
    ks = [k.strip() for k in m.group(1).split(",")]
    return "\\cite{" + ",".join(sorted(ks, key=lambda k: number[k])) + "}"


new_head = re.sub(r"\\cite\{([^}]*)\}", fix_group, head)

shutil.copyfile(PAPER, PAPER + ".prebibsort")
with open(PAPER, "w", encoding="utf-8", newline="") as f:
    f.write(new_head + MARK + new_bib + r"\end{thebibliography}" + after)

print(f"reordered {len(order)} bibliography entries; backup at {PAPER}.prebibsort")
