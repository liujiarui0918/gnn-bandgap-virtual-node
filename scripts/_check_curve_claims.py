"""Check the training-curve claims written into the supplementary text."""

import json

MODELS = {
    "T1": "T1_CGCNN_baseline",
    "T2": "T2_ALIGNN_baseline",
    "T3": "T3_M3GNet_baseline",
    "T4": "T4_ALIGNN_plain_single_virtualnode",
    "T7": "T7_ALIGNN_huber_virtualnode",
    "T10": "T10_ALIGNN_decay_huber_virtualnode",
}
BASE = {
    "MP": "results/mp_hzw_relaxed_clean_equal_atoms_mp_t",
    "HZW": "results/mp_hzw_relaxed_clean_equal_atoms_hzw_t",
}


def hist(src, mid):
    with open(f"{BASE[src]}/{MODELS[mid]}/history.json", encoding="utf-8") as f:
        return json.load(f)


print("== best-validation epoch (val_mae argmin) ==")
for src in BASE:
    row = []
    for mid in MODELS:
        h = hist(src, mid)
        b = min(h, key=lambda r: r["val_mae"])
        row.append(f"{mid}={b['epoch']}")
    print(f"  {src:3s} " + "  ".join(row))

print("\n== generalization gap in MAE (val_mae - train_mae), ALIGNN variants ==")
for src in BASE:
    print(f"  -- {src}")
    for mid in ["T2", "T4", "T7", "T10"]:
        h = hist(src, mid)
        g = {r["epoch"]: r["val_mae"] - r["train_mae"] for r in h}
        print(
            "     %-4s ep10=%.4f ep20=%.4f ep25=%.4f ep30=%.4f ep40=%.4f ep50=%.4f  (ep50-ep25=%+.4f)"
            % (mid, g[10], g[20], g[25], g[30], g[40], g[50], g[50] - g[25])
        )

print("\n== M3GNet (T3) validation MAE tail ==")
for src in BASE:
    h = hist(src, "T3")
    vm = {r["epoch"]: r["val_mae"] for r in h}
    print(
        f"  {src:3s} "
        + " ".join(f"ep{e}={vm[e]:.4f}" for e in (25, 30, 35, 40, 41, 43, 45, 50))
    )
    best = min(h, key=lambda r: r["val_mae"])
    print(
        f"      best ep{best['epoch']} = {best['val_mae']:.4f}; last-10-epoch spread = "
        f"{max(vm[e] for e in range(41, 51)) - min(vm[e] for e in range(41, 51)):.4f}"
    )

print("\n== ALIGNN-variant val MAE, last 10 epochs spread ==")
for src in BASE:
    for mid in ["T2", "T4", "T7", "T10"]:
        h = hist(src, mid)
        vm = {r["epoch"]: r["val_mae"] for r in h}
        tail = [vm[e] for e in range(41, 51)]
        print(
            f"  {src:3s} {mid:4s} min={min(tail):.4f} max={max(tail):.4f} spread={max(tail) - min(tail):.4f}"
        )
