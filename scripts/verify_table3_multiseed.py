# One-shot verification: Table 3 values in the paper vs direct metrics.json computation.
import json
import pathlib
import re
import statistics as st

root = pathlib.Path("results")
models = {
    "T1": "T1_CGCNN_baseline",
    "T2": "T2_ALIGNN_baseline",
    "T3": "T3_M3GNet_baseline",
    "T4": "T4_ALIGNN_plain_single_virtualnode",
    "T7": "T7_ALIGNN_huber_virtualnode",
    "T10": "T10_ALIGNN_decay_huber_virtualnode",
}
calc = {}
for src in ("mp", "hzw"):
    for tag, m in models.items():
        for metric in ("mae", "rmse", "metal_f1"):
            vals = []
            for suf in ("", "_seed1042", "_seed2042"):
                f = root / f"mp_hzw_relaxed_clean_equal_atoms_{src}_t{suf}" / m / "metrics.json"
                vals.append(json.loads(f.read_text())[metric])
            calc[(src, tag, metric)] = (st.mean(vals), st.stdev(vals))

tex = pathlib.Path("gnn_bandgap_research_paper_draft_en.tex").read_text(encoding="utf-8")
pair_re = re.compile(r"([\d.]+)\s*\\,\{\\pm\}\\,\s*([\d.]+)")
errors = 0
for tag in models:
    line = next((l for l in tex.splitlines() if l.startswith(f"{tag} & ") and "pm" in l), None)
    if line is None:
        print(f"{tag}: row not found")
        errors += 1
        continue
    nums = pair_re.findall(line)
    if len(nums) != 6:
        print(f"{tag}: expected 6 pm pairs, got {len(nums)}")
        errors += 1
        continue
    keys = [("mp", "mae"), ("mp", "rmse"), ("hzw", "mae"), ("hzw", "rmse"), ("mp", "metal_f1"), ("hzw", "metal_f1")]
    for (mean_s, std_s), (src, metric) in zip(nums, keys):
        cm, cs = calc[(src, tag, metric)]
        if abs(float(mean_s) - round(cm, 4)) > 5e-5 or abs(float(std_s) - round(cs, 4)) > 5e-5:
            print(f"MISMATCH {tag} {src} {metric}: paper {mean_s}+-{std_s} vs calc {cm:.4f}+-{cs:.4f}")
            errors += 1
print("ALL_TABLE3_VALUES_VERIFIED" if errors == 0 else f"{errors} problems")
