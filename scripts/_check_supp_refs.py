"""Cross-check every hard-coded 'Supplementary Table/Figure/Section SN' reference
in the main text against the actual numbering of the supplementary document.

Numbers come from the supplementary .aux (\\newlabel), captions/titles from its
.tex, so the mapping is whatever LaTeX really produced.
"""

import re

SUPP_TEX = "gnn_bandgap_research_supplementary_en.tex"
SUPP_AUX = "gnn_bandgap_research_supplementary_en.aux"
PAPER = "gnn_bandgap_research_paper_draft_en.tex"

supp = open(SUPP_TEX, encoding="utf-8").read()
aux = open(SUPP_AUX, encoding="utf-8").read()

# label -> printed number
label_num = {}
for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux):
    label_num[m.group(1)] = m.group(2)


def strip(s):
    s = re.sub(r"\\(textnormal|emph|textbf|mathrm|text)\{([^{}]*)\}", r"\2", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    return re.sub(r"[{}\\]", "", s).strip()


# environment -> (label, first sentence of caption)
items = {"table": {}, "figure": {}}
for env in items:
    for m in re.finditer(
        r"\\begin\{" + env + r"\}(.*?)\\end\{" + env + r"\}", supp, re.S
    ):
        blk = m.group(1)
        lab = re.search(r"\\label\{([^}]*)\}", blk)
        cap = re.search(r"\\caption\{(.*?)\}\s*\n", blk, re.S)
        if lab:
            text = strip(cap.group(1))[:95] if cap else "(no caption)"
            items[env][label_num.get(lab.group(1), "?")] = (lab.group(1), text)

# sections in document order
sections = []
for i, m in enumerate(re.finditer(r"\\section\{(.*?)\}\s*\n", supp)):
    sections.append(("S%d" % (i + 1), strip(m.group(1))))

print("=== SUPPLEMENTARY TABLES ===")
for k in sorted(items["table"], key=lambda x: int(x.lstrip("S") or 0)):
    print(f"  {k:4s} {items['table'][k][1]}")
print("\n=== SUPPLEMENTARY FIGURES ===")
for k in sorted(items["figure"], key=lambda x: int(x.lstrip("S") or 0)):
    print(f"  {k:4s} {items['figure'][k][1]}")
print("\n=== SUPPLEMENTARY SECTIONS ===")
for num, title in sections:
    print(f"  {num:4s} {title}")

print("\n=== MAIN-TEXT REFERENCES (with context) ===")
paper = open(PAPER, encoding="utf-8").read()
paper_body = paper.split(r"\begin{thebibliography}")[0]
seen = set()
for m in re.finditer(
    r"Supplementary (Table|Figure|Section)s?~?(S\d+)((?:\s*(?:and|,)~?S\d+)*)",
    paper_body,
):
    kind = m.group(1)
    nums = [m.group(2)] + re.findall(r"S\d+", m.group(3) or "")
    ctx = re.sub(r"\s+", " ", paper_body[max(0, m.start() - 95) : m.end() + 25])
    for n in nums:
        key = (kind, n)
        target = {"Table": items["table"], "Figure": items["figure"]}.get(kind)
        if target is not None:
            resolved = target.get(n, ("?", "*** NO SUCH ITEM ***"))[1]
        else:
            resolved = dict(sections).get(n, "*** NO SUCH SECTION ***")
        flag = "  " if "NO SUCH" not in resolved else "XX"
        print(f"{flag} {kind} {n}: {resolved}")
        if key not in seen:
            print(f"      ctx: ...{ctx}...")
            seen.add(key)
