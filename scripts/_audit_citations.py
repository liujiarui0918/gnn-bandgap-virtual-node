"""One-off audit: abstract length, citation completeness, first-citation ordering."""

import re
import sys

PAPER = "gnn_bandgap_research_paper_draft_en.tex"

tex = open(PAPER, encoding="utf-8").read()
abst = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
print("abstract words:", len(abst.split()), " chars:", len(abst.strip()))

body, bib = tex.split(r"\begin{thebibliography}")
order, seen = [], set()
for m in re.finditer(r"\\cite\{([^}]*)\}", body):
    for k in [k.strip() for k in m.group(1).split(",")]:
        if k not in seen:
            seen.add(k)
            order.append(k)
defined = re.findall(r"\\bibitem\{([^}]*)\}", bib)
print("cited:", len(order), " defined:", len(defined))
print("cited-but-undefined:", [k for k in order if k not in defined])
print("defined-but-uncited:", [k for k in defined if k not in seen])

pos = {k: i + 1 for i, k in enumerate(defined)}
seq = [pos[k] for k in order if k in pos]
print("\nnumber sequence in first-citation order:")
print(seq)
bad = [(i, seq[i - 1], seq[i]) for i in range(1, len(seq)) if seq[i] < seq[i - 1]]
print("out-of-order steps:", len(bad))
for i, a, b in bad[:15]:
    print("   [%d] then [%d]   (%s -> %s)" % (a, b, order[i - 1], order[i]))

# also flag \cite groups whose printed numbers are not ascending
print("\nnon-ascending \\cite groups:")
for m in re.finditer(r"\\cite\{([^}]*)\}", body):
    ks = [k.strip() for k in m.group(1).split(",")]
    ns = [pos.get(k, 0) for k in ks]
    if ns != sorted(ns):
        print("  ", ns, ks)
