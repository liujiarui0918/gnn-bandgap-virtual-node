"""Verify every numeric claim in the paper/SI that is backed by an analysis CSV."""

import csv

ok = 0
bad = 0


def load(path):
    rows = [r for r in open(path, encoding="utf-8") if not r.lstrip().startswith("#")]
    return list(csv.DictReader(rows))


def check(label, got, want, tol=5e-5):
    global ok, bad
    if got is None:
        print(f"  ?? {label}: NOT FOUND (paper says {want})")
        bad += 1
        return
    good = abs(got - want) <= tol
    print(f"  {'OK' if good else 'XX'} {label}: csv={got:+.4f} paper={want:+.4f}")
    ok += good
    bad += not good


def tcode(s):
    return s.split("_")[0].upper()


print("=== paired bootstrap (Supplementary Table S7 + main-text Section 5.1) ===")
pb = load("results/analysis/paired_bootstrap.csv")
PAPER_PB = {
    ("mp", "T4", "T2"): (-0.0075, -0.0121, -0.0028),
    ("mp", "T7", "T4"): (-0.0076, -0.0122, -0.0029),
    ("mp", "T7", "T2"): (-0.0151, -0.0198, -0.0103),
    ("mp", "T10", "T7"): (-0.0012, -0.0058, 0.0033),
    ("mp", "T10", "T4"): (-0.0088, -0.0135, -0.0043),
    ("hzw", "T4", "T2"): (-0.0104, -0.0147, -0.0060),
    ("hzw", "T7", "T4"): (-0.0078, -0.0120, -0.0035),
    ("hzw", "T7", "T2"): (-0.0181, -0.0224, -0.0138),
    ("hzw", "T10", "T7"): (0.0030, -0.0011, 0.0071),
    ("hzw", "T10", "T4"): (-0.0047, -0.0094, -0.0001),
}
for (src, a, b), (d, lo, hi) in PAPER_PB.items():
    row = next(
        (
            r
            for r in pb
            if r["label_source"].lower() == src
            and tcode(r["model_a"]) == a
            and tcode(r["model_b"]) == b
        ),
        None,
    )
    if row is None:
        print(f"  ?? {src} {a}-{b}: row not found")
        bad += 1
        continue
    check(f"{src} {a}-{b} delta ", float(row["delta_mae"]), d)
    check(f"{src} {a}-{b} CI lo ", float(row["ci_low"]), lo)
    check(f"{src} {a}-{b} CI hi ", float(row["ci_high"]), hi)

print("\n=== cross-label evaluation (Supplementary Table S8, T10-T7 sign pattern) ===")
cl = load("results/analysis/cross_label_eval.csv")


def clmae(train, ev, mid):
    r = next(
        (
            r
            for r in cl
            if r["train_source"].lower() == train
            and r["eval_source"].lower() == ev
            and tcode(r["model"]) == mid
        ),
        None,
    )
    return float(r["mae"]) if r else None


for (tr, ev), want in {
    ("mp", "mp"): -0.0012,
    ("mp", "hzw"): -0.0024,
    ("hzw", "mp"): 0.0025,
    ("hzw", "hzw"): 0.0030,
}.items():
    a, b = clmae(tr, ev, "T10"), clmae(tr, ev, "T7")
    check(f"train={tr:3s} eval={ev:3s} T10-T7", (a - b) if (a and b) else None, want)

print("\n=== strict-subset gains (Supplementary Table S10 + main-text Section 6.1) ===")
ss = load("results/analysis/strict_subset_metrics.csv")


def smae(src, subset, mid):
    r = next(
        (
            r
            for r in ss
            if r["label_source"].lower() == src
            and r["subset"] == subset
            and tcode(r["model"]) == mid
        ),
        None,
    )
    return float(r["mae"]) if r else None


for src, subset, a, b, want in [
    ("mp", "clean", "T4", "T2", -0.0081),
    ("mp", "noisy", "T4", "T2", 0.0015),
    ("mp", "clean", "T7", "T4", -0.0089),
    ("mp", "noisy", "T7", "T4", 0.0140),
    ("hzw", "clean", "T4", "T2", -0.0101),
    ("hzw", "noisy", "T4", "T2", -0.0141),
    ("hzw", "clean", "T7", "T4", -0.0071),
    ("hzw", "noisy", "T7", "T4", -0.0186),
]:
    x, y = smae(src, subset, a), smae(src, subset, b)
    check(f"{src:3s} {subset:5s} {a}-{b}", (x - y) if (x and y) else None, want)

print("\n=== over-smoothing >80-atom deltas (main-text Table 4 vs CSV) ===")
osc = load("results/analysis/oversmoothing_delta_ci_full.csv")
print("   columns:", list(osc[0].keys()))
PAPER_OS = {
    ("mp", "T4", "cosine"): -0.087,
    ("mp", "T7", "cosine"): -0.087,
    ("mp", "T10", "cosine"): -0.086,
    ("hzw", "T4", "cosine"): -0.091,
    ("hzw", "T7", "cosine"): -0.090,
    ("hzw", "T10", "cosine"): -0.093,
    ("mp", "T4", "variance"): 0.277,
    ("mp", "T7", "variance"): 0.233,
    ("mp", "T10", "variance"): 0.164,
    ("hzw", "T4", "variance"): 0.270,
    ("hzw", "T7", "variance"): 0.160,
    ("hzw", "T10", "variance"): 0.072,
    ("mp", "T4", "dirichlet"): 107.4,
    ("mp", "T7", "dirichlet"): 83.7,
    ("mp", "T10", "dirichlet"): 46.6,
    ("hzw", "T4", "dirichlet"): 100.1,
    ("hzw", "T7", "dirichlet"): 42.5,
    ("hzw", "T10", "dirichlet"): -4.9,
}
keys = list(osc[0].keys())
src_k = "label_source" if "label_source" in keys else "source"
sub_k = "subset" if "subset" in keys else None
val_k = next((k for k in keys if "mean" in k), None)
for (src, mid, metric), want in PAPER_OS.items():
    r = next(
        (
            r
            for r in osc
            if r[src_k].lower() == src
            and tcode(r["model"]) == mid
            and r["metric"].lower().startswith(metric[:6])
            and (sub_k is None or r[sub_k] in ("gt80", ">80"))
        ),
        None,
    )
    tol = 0.05 if metric == "dirichlet" else 5e-4
    check(f"{src:3s} {mid:3s} {metric:9s}", float(r[val_k]) if r else None, want, tol)

print(f"\nTOTAL: {ok} pass / {bad} fail")
