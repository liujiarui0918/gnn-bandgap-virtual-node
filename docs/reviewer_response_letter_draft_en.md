# Response to Reviewers — DRAFT (all analyses and experiments complete; DFT footnote pending provider info)

**Manuscript:** Robust Global Virtual-Node Graph Neural Networks for Crystal Band-Gap Prediction with Cross-Source Label Consistency Analysis

**Status of this document:** All ten comments are now fully backed by completed work — the earlier analyses (paired bootstrap, cross-label evaluation, cohort profiling, global-broadcast decomposition, strict-subset re-scoring, F1 thresholds) and every follow-up GPU experiment (multi-seed 24 runs; full-test-set oversmoothing rerun; strict-Huber S14; M3GNet tuning probes; Huber-β and decay-floor sensitivity sweeps). No external confirmations remain open except the non-blocking DFT code/functional footnote for the HZW provider (Comment 7). Identifiers are finalized against the revised manuscript and Supplementary Information (SI): **Table 1** is the data-audit table (now split into two panels), **Table 2** the model-variant definitions, **Table 3** the main results table (T-series MAE/RMSE/F1, three-seed mean ± std with bootstrap annotations), and **Table 4** the oversmoothing diagnostics (full-test-set values); the main figures are **Fig. 3** (main results), **Fig. 4** (mechanism diagnostics), and **Fig. 5** (element/symmetry-resolved analysis), and the new SI items are **Tables S6–S16** and **Fig. S4** (referenced per response below).

**Legend for progress tags** (shown at the end of each response):
- `[DONE]` — completed and reflected in the revised manuscript / SI.
- `[RUNNING]` / `[PENDING GPU]` — historical tags; no item carries them any longer (all GPU batches P0/P1/P2 have finished).

---

## Cover note to the Editor and Reviewers

We thank the Editor and the Reviewers for their careful and constructive assessment. The reviews raised ten points spanning four themes: (i) statistical rigor of the model ranking (Comments 1, 3), (ii) fairness and transparency of baselines and hyperparameters (Comments 2, 10), (iii) data-grounded support for our mechanistic claims (Comments 4, 5, 6), and (iv) terminology, threshold definitions, and citations (Comments 7, 8, 9).

We wish to state at the outset that none of the ten comments overturns the paper's central conclusion — that *a single plain virtual node combined with a Huber objective is the most robust design across label sources*. Rather, they identify places where our evidence and exposition must be strengthened. In several cases the reviewers correctly anticipated the limits of a single-seed comparison; the revised abstract and Section 7.2 already temper the ranking claims accordingly — the paired bootstrap confirms that the decay variant and the Huber-only virtual-node variant are statistically inseparable in MAE on MP (see Comment 1). We have reorganized the revision around multi-seed statistics, data-grounded mechanism analyses, and rigorous terminology, and we detail each below.

Reviewer comments are reproduced in italics, followed by our response, the location of the change in the revised manuscript, and the new evidence introduced.

---

## Comment 1 — Single seed, no statistical significance

> *The reported MAE differences among the leading variants are only on the order of 0.001–0.008 eV, obtained from a single random seed and a single data split. Such differences cannot support the claimed ranking. Please report at least three seeds with mean ± standard deviation in Table 3 and Fig. 3.*

**Response.** We fully accept this point, which also matches a limitation we had ourselves noted. In the revision we address it on two complementary axes that separate two distinct sources of uncertainty:

1. *Test-set sampling noise* (orthogonal to training randomness): we add paired bootstrap 95% confidence intervals (B = 10,000; bootstrap seed 20260714) for ten model pairs — five per label source across the T2→T4→T7→T10 ladder (adjacent and cumulative), including the plain-ALIGNN baseline — reporting the ΔMAE interval and P(Δ < 0). This is available from existing predictions and is reported immediately.
2. *Training randomness*: we have now retrained the six core reported models with three seeds (existing seed 42 plus two new seeds, 1042 and 2042), across both label sources, holding the train/val/test split fixed so that only initialization and shuffling vary — 6 models × 2 label sources × 3 seeds = 24 completed runs. The two new seeds used validation-MAE early stopping (patience 8) to bound compute; the best-validation checkpoint convention is unchanged, so the metric protocol is identical to the original runs. The main results table (Table 3) and Fig. 3 are updated to three-seed mean ± standard deviation, and we report per-seed direction consistency rather than an underpowered n = 3 significance test. Both robust ladder steps are direction-consistent across all three seeds on both label sources (3/3): the virtual-node gain (T4−T2) is −0.0080 ± 0.0006 eV on MP and −0.0104 ± 0.0011 eV on HZW, and the Huber gain (T7−T4) is −0.0088 ± 0.0013 eV on MP and −0.0086 ± 0.0009 eV on HZW (seed means ± std). The decay step (T10−T7) is small and sign-unstable — −0.0004 ± 0.0017 eV on MP (2/3 same sign) and +0.0018 ± 0.0023 eV on HZW (2/3 same sign) — so the multi-seed replication and the paired bootstrap independently reach the same verdict that T10 and T7 are inseparable, and T10 is downgraded per the pre-registered rule.

**Pre-registered narrative (honest ranking), now resolved by the bootstrap and confirmed by seed replication.** We committed, before seeing the multi-seed results, to a fixed writing rule; the paired bootstrap already settles the key question, and the completed multi-seed retraining adds the orthogonal check and reaches the same verdict. Both robust ladder steps survive on both sources. The virtual-node gain (**ALIGNN+VN over ALIGNN**) is significant on both: ΔMAE = −0.0075 eV on MP (95% CI [−0.0121, −0.0028], P(Δ<0) = 0.9995) and −0.0104 eV on HZW (95% CI [−0.0147, −0.0060], P(Δ<0) = 1.0000). The Huber gain (**ALIGNN+VN+Huber over ALIGNN+VN**) is likewise significant on both: −0.0076 eV on MP (95% CI [−0.0122, −0.0029], P(Δ<0) = 0.9994) and −0.0078 eV on HZW (95% CI [−0.0120, −0.0035], P(Δ<0) = 0.9997); the cumulative ALIGNN+VN+Huber-over-ALIGNN gap is significant as well (−0.0151 on MP, −0.0181 on HZW, both P(Δ<0) = 1.0000). **The virtual-node gain and the Huber gain are thus each bootstrap-backed; the only step without such backing is decay (T10−T7), non-significant on both sources.** The *high-risk* claim was the MP-side ordering of the decay variant versus the Huber-only variant, whose single-seed gap is only −0.0012 eV. **The bootstrap confirms the two are statistically indistinguishable on MP — ΔMAE = −0.0012 eV, 95% CI [−0.0058, +0.0033], P(Δ<0) = 0.70 — and on HZW the point estimate even favors the Huber-only model (ΔMAE = +0.0030 eV, 95% CI [−0.0011, +0.0071], P(Δ<0) = 0.07). We therefore state plainly that the two are inseparable in MAE, that any residual value of broadcast decay lies in the reduced systematic bias (mean residual 0.025 → 0.014 eV on MP) and RMSE (0.504 → 0.497 on MP) observed in the seed-42 run rather than in MAE, and we foreground the Huber-only virtual-node model as the recommended cross-source design.** This strengthens, rather than weakens, the paper's conservative main thesis. The abstract and Section 7.2 wording have been softened in lockstep.

**Location in revised manuscript.** Abstract (ranking wording, already softened); Section 5 (Experimental Setup: seed protocol and statistical methodology, added — including the two new seeds' validation-MAE early-stopping protocol, verbatim-identical cross-seed test-set sample ids, and the note that only M3GNet triggered early stopping); Section 6, Table 3 (revised, three-seed mean ± std) and Fig. 3 (revised); Limitations (multi-seed item rewritten); SI ("Multi-seed replication" per-seed detail table; paired-bootstrap Table S7).

**New evidence.** Table 3 (revised) with three-seed mean ± std; Fig. 3 (revised) with error bars and per-seed scatter; the "Multi-seed replication" per-seed metrics table (24 runs); Table S7 (paired bootstrap CIs, all ten pairs — five per label source across the T2→T4→T7→T10 ladder, adjacent and cumulative).

**Progress.** Paired bootstrap CIs (ten pairs, both sources, baseline included) `[DONE]`. Multi-seed retraining (24 runs: 6 core models × 2 label sources × 3 seeds 42/1042/2042) `[DONE]`.

---

## Comment 2 — M3GNet baseline unexpectedly weak

> *The M3GNet baseline (MP MAE ~0.286) is markedly worse than ALIGNN (~0.240) with no explanation. This raises concern that the baseline is not tuned or is unfairly configured.*

**Response.** We partly accept this and have clarified the facts and added a fairness check. (a) M3GNet here is trained *from scratch* via the matgl implementation, not fine-tuned from a pretrained potential; (b) it uses a modest default-scale configuration (3 blocks, 128 units) with no task-level tuning; (c) M3GNet was designed primarily as a universal interatomic potential emphasizing many-body geometry, and single-fidelity band-gap regression is not its target task — public leaderboards (e.g., Matbench `matbench_mp_gap`, JARVIS) consistently place ALIGNN above most message-passing models on gap tasks. To test whether the gap is a tuning artifact, we ran three single-factor tuning probes on MP labels — doubling the budget to 100 epochs, halving the learning rate to 5×10⁻⁴, and doubling the width to 256 units. None closes the gap: the best probe (100 epochs) reaches MAE 0.2867, within the baseline's seed band (0.2897 ± 0.0033), the lower learning rate is neutral-to-worse (0.2912), and the wider model overfits and stops early at a clearly worse 0.3112 — versus 0.2400 for the ALIGNN baseline. The ~0.05 eV deficit therefore does not appear to be a tuning artifact under single-factor probes: it does not disappear under a larger budget, a gentler learning rate, or more capacity (single-seed probes, joint search out of scope).

**Pre-registered narrative.** We committed: if limited tuning improves M3GNet by < 0.01 eV, we write that it remains behind under these limited probes, which argues against gross under-fitting (single-seed probes, joint search out of scope); if it improves substantially, we add a "tuned" row and update the baseline statement honestly. The observed outcome is the first branch — the best probe improves nothing beyond seed noise — and the manuscript states this. Either outcome leaves the VN/Huber thesis (built on the ALIGNN → +VN → +Huber ladder) untouched.

**Location in revised manuscript.** Section 4.2 / Section 5 (full M3GNet training configuration disclosure, merged into the unified training-configuration table); Section 7 (the "On the M3GNet baseline" paragraph now cites the tuning probes); SI Table S16 (tuning-probe results).

**New evidence.** Training-configuration table (Table S14, shared with Comment 10); Table S16 (M3GNet single-factor tuning probes with best/ran epochs); leaderboard comparison citations.

**Progress.** Configuration disclosure and literature comparison `[DONE]`. Limited tuning runs (3 runs, batch P2) `[DONE]` — no probe closes the gap; conclusion "the gap does not appear to be a tuning artifact under single-factor probes (single-seed probes, joint search out of scope)" written into Section 7 and SI Table S16.

---

## Comment 3 — Oversmoothing diagnostic uses only 240 samples

> *The mechanistic evidence in Section 7.5 relies on only 240 stratified samples, which is unconvincing. Please rerun on the full test set (8,455) or rigorously justify the sampling and add error bars.*

**Response.** We fully accept this. The 240-sample figure was an artifact of running the diagnostic on a CPU login node; the diagnostic itself is inference-level, and we have now rerun it on the full test set (8,455 samples per label source, all reported variants, both sources), replacing the 240-sample numbers as the primary evidence; the 240-sample version is retained in the SI text as a cross-check of sampling representativeness (it agrees with the full-set version in the direction of every significant effect). We also added bootstrap 95% CIs throughout and documented the stratified pilot protocol (atom-count-bucket rotation with quota). The full-set result *strengthens* the size-conditional pattern: on the >80-atom subset (now n = 422 instead of 78) the plain virtual-node model and its Huber variant both show the complete relief signature — lower final-layer mean pairwise cosine, higher node-feature variance, and higher Dirichlet energy — on both label sources with all 95% CIs excluding zero (at the pilot scale the Huber variant's Dirichlet CI still crossed zero on HZW; at full scale it no longer does). The decay variant matches them on cosine and variance, with only its HZW Dirichlet interval still crossing zero. The effect remains genuinely size-specific: on the full panel the global-token models raise variance and Dirichlet energy everywhere, but their mean pairwise cosine is higher rather than lower, so we continue to claim a size-conditional — not universal — over-smoothing benefit. We additionally added a metric caveat to Section 7.6 and the captions of Table 4 and Table S12: of the three diagnostics only the pairwise cosine similarity is invariant to the overall embedding scale, so we treat the large-structure cosine reduction as the primary evidence and the variance and Dirichlet-energy increases as corroborating but norm-sensitive.

**Pre-registered narrative.** We committed: if the full-set result preserves the size-conditional signal, we keep the conditional claim and upgrade it to full-set evidence; if it weakens, we downgrade. The observed outcome is the first branch — the signal is preserved and sharpened — and Section 7.6, Table 4, and Fig. 4(c,d) now report full-set values.

**Location in revised manuscript.** Section 7.6 and Table 4 (revised, full-set values, n = 422 large-structure subset); Fig. 4(c,d) (regenerated from the full-set diagnostic); SI Table S12 (full-set bootstrap CIs, all 36 model × subset × metric cells) with the 240-sample pilot noted as a consistency check.

**New evidence.** Table 4 (revised, full-set); Fig. 4(c,d) (revised); Table S12 (full-set 95% CIs; pilot agreement statement).

**Progress.** Error bars `[DONE]`. Full-test-set rerun `[DONE]` — signal preserved and strengthened; all manuscript artifacts updated.

---

## Comment 4 — Missing verification on a globally dependent property (formation energy / dielectric)

> *The band gap is a fairly localized chemical property, so it is weak evidence for a "global virtual node." Consider adding a more globally dependent target such as formation energy or dielectric constant.*

**Response.** We respectfully argue that this extension, while valuable, lies outside the deliberate scope of the paper, and we address it with an explicit scoping statement rather than a new experiment. (a) The paper's scope is intentionally limited to the band-gap task because its core contribution axis is the *dual-label cross-source design*: every model is trained and evaluated on the same structures under two independent label sources (MP and HZW), which is what enables the label-consistency audit and the robustness conclusions. (b) The HZW database provides no second-property labels of any kind (we verified that neither our MP fetch cache nor the HZW raw download contains formation-energy or dielectric values). A formation-energy experiment could therefore only be run single-source on MP labels — it would have to abandon precisely the cross-source dimension that carries the paper's main claims, and a single-source result would neither test nor strengthen them. (c) The first item of the revised Limitations now states this scope explicitly: the conclusions are claimed for band-gap prediction under cross-source labels, not for globally dependent properties in general; within that scope, the size-conditional representation diagnostics (Section 7.6) are positioned as the paper's in-scope mechanistic evidence for the global token. (d) For the value of virtual nodes on strongly global properties we rely on the literature rather than re-verification: Okabe et al. (VGNN, Nature Computational Science 2024) already demonstrate virtual-node GNNs on full-phonon-spectrum prediction, and we cite this directly. (e) Finally, the revision adds an element- and symmetry-resolved error analysis (a new subsection plus a new Figure 5: a periodic-table heatmap of per-element MAE, a per-element T7−T2 improvement heatmap, MAE by crystal system, and MAE binned by band gap), which supplies physically resolved evidence within the band-gap target itself and thereby partly addresses the "physical insight" concern.

**Location in revised manuscript.** Section 8 Limitations (first item: explicit scope statement); Section 2 / Section 7 (Okabe et al. VGNN citation for the global-property argument); Section 7.6 (size-conditional diagnostics positioned as in-scope mechanism evidence).

**New evidence.** Not applicable (scoping statement and literature argument; no new experiment).

**Progress.** `[DONE — addressed by scoping statement, no new experiment]`.

---

## Comment 5 — Decay divergence explained only by conjecture

> *The claim in Section 7.2 that "HZW needs sustained global calibration" is speculative. Please provide data-grounded evidence, e.g., identify the HZW crystals harmed by decay and analyze their features.*

**Response.** We fully accept this, and it is the highest-return revision item: four of the five supporting analyses require no new compute and use data already available locally. We replace the single conjectural interpretation with an evidence list:

1. **Training-dynamics evidence** — per-epoch validation-MAE curves for the Huber-only and decay variants on both sources, with the decay window annotated and the per-epoch (decay − Huber) difference overlaid. The curves show HZW degradation coincides with the deep-decay stage: on HZW the (decay − Huber) validation-MAE gap opens to roughly +0.005 to +0.008 eV inside the deep-decay window and keeps climbing in the later epochs, whereas on MP it merely oscillates around zero.
2. **Cross-label evaluation** — a 2×2 (training label × evaluation label) matrix, exploiting the identical structures and split, to decompose whether the harm from decay originates in the training signal or in the evaluation labels. The matrix shows the sign of the decay effect is fixed by the training source, not the evaluation label: the MP-trained decay model improves MAE under both evaluation labels (0.2249 → 0.2237 on MP labels, 0.2314 → 0.2290 on HZW labels), while the HZW-trained decay model worsens it under both (0.1940 → 0.1970 on HZW labels, 0.2300 → 0.2325 on MP labels). These are seed-42 checkpoints, and each of the four individual deltas lies within the paired-bootstrap noise band, so the evidence is the consistent sign pattern across the 2×2 matrix rather than any single comparison; the revised Section 7.2 and SI Table S8 now state this caveat explicitly. Decay's effect is therefore a training-dynamics phenomenon, not an evaluation-label artifact.
3. **Harmed-cohort profiling** — the worst-decile vs best-decile HZW crystals by decay-induced error change (846 crystals per decile of the 8,455-sample test set), compared on element families, crystal system / space group, gap × atom-count buckets, and MP−HZW label difference, with bootstrap CIs on proportion differences. Four dimensions separate the harmed from the benefited decile with 95% bootstrap CIs excluding zero: the harmed crystals over-represent transition-metal compounds (+4.8 pp, CI [+0.1, +9.3]) and under-represent tetragonal systems (−2.8 pp), medium-symmetry space groups 75–167 (−4.4 pp), and small gaps in (0, 0.5) eV (−3.2 pp). We state candidly that the separation is weak — most dimensions are not individually significant and the four significant CIs sit close to zero — so we present this as a suggestive, not decisive, cohort signature.
4. **Global-broadcast decomposition** — decomposing the Huber-only vs decay checkpoints by error component confirms decay acts as a training-time global-broadcast schedule (strong early, attenuated late): on MP it lowers systematic over-prediction (mean residual 0.025 → 0.014 eV) and RMSE (0.504 → 0.497) with MAE unchanged within noise, whereas on HZW the same late-stage attenuation lowers bias but raises RMSE (0.446 → 0.454) and MAE — evidence consistent, in the seed-42 run, with HZW predictions depending on sustained late-stage global calibration that decay removes.
5. **Dose–response (completed)** — a sweep of the decay floor (min_scale ∈ {0.5, 0.75} vs the default 0.25), shared with Comment 10. On HZW labels the MAE recovers monotonically as decay is weakened: 0.1970 (floor 0.25) → 0.1950 (0.5) → 0.1943 (0.75) → 0.1940 (no decay, the Huber-only model). This dose–response pattern independently indicates that the decay schedule itself — not any particular floor value — is what degrades the HZW MAE in these single-seed runs, completing the mechanism argument (SI Table S15).

**Location in revised manuscript.** Section 7.2 (already rewritten from "one plausible interpretation" to a three-evidence, data-indexed argument — cross-label, training dynamics, cohort profiling); SI decay-mechanism section (Table S8 cross-label matrix, Table S9 cohort profile, Fig. S4 training dynamics, plus the global-broadcast decomposition).

**New evidence.** Fig. S4 (training dynamics); Table S8 (2×2 cross-label matrix); Table S9 (cohort profile); the global-broadcast error decomposition (SI decay-mechanism section); Table S15 (dose–response over the decay floor, showing monotone HZW recovery towards the no-decay limit).

**Progress.** Four zero-cost analyses — training dynamics, cross-label evaluation, cohort profiling, and the global-broadcast decomposition — `[DONE]`. Dose–response floor sweep (4 runs, batch P1) `[DONE]` — HZW MAE recovers monotonically as decay weakens.

---

## Comment 6 — Relaxed-set label noise may mask architectural gains

> *The relaxed setting retains same-class band-gap disagreements above 1 eV, so label noise could obscure the architectural gains. Please compare the leading variants on a strictly consistent subset.*

**Response.** We accept this, and much of the required asset already exists (a strict-clean lineage trained under matched metal/non-metal class, |Δgap| ≤ 0.5 eV, and volume difference ≤ 8%). We respond on two complementary levels: (a) an immediate, zero-cost *evaluation-side* test — re-scoring the leading variants on the clean vs noisy test subset under the strict thresholds, on both label sources; and (b) a *training-side* test — completing the strict-clean training triad by adding the missing Huber variant so the reviewer's requested comparison is available. We are careful to note that strict and relaxed sets differ in sample count and test set, so we compare *relative gains within each setting* and never compare absolute MAE across settings. On the strict-clean test subset (n = 7,958) the Huber-over-VN gain is preserved on both sources (−0.0089 eV on MP, −0.0071 eV on HZW); on the tiny strict-noisy complement (n = 497) the picture is high-variance and source-dependent (the gain reverses to +0.0140 eV on MP but enlarges to −0.0186 eV on HZW). The baseline contrast answers the reviewer's masking question directly: the virtual-node architectural gain (ALIGNN+VN over ALIGNN) on MP is −0.0081 eV on the clean subset but +0.0015 eV on the noisy complement — that is, on MP the architectural gain is visible only once inconsistent labels are removed, exactly the masking the reviewer suspected — whereas on HZW the direction is opposite (−0.0101 clean vs −0.0141 noisy), so the masking effect is source-dependent rather than universal. The evaluation-side test therefore argues against the leading-variant gain being an artifact of label noise — it is at least as clean on the strictly consistent subset. The training-side test is now also complete: retraining the virtual-node model with the Huber loss directly on the strict-clean dataset (S14, the Huber counterpart of the existing MSE-loss S4 run; identical protocol) improves the strict test MAE from 0.2001 to 0.1936 (n = 8,154). The strict-training Huber gain (−0.0064 eV) matches the relaxed-lens T4→T7 gain (−0.0076 eV on MP) in size, so the Huber improvement survives when the noise channel is removed by construction rather than by post-hoc filtering — a single strict-lens run with one seed, like its S4 counterpart, so it argues against, though cannot definitively exclude, the noise-absorption explanation (hedged accordingly in Section 7.1 and the SI strict-subset section).

**Pre-registered narrative.** If the architectural gain is larger on the clean subset, this supports "noise partly masks architectural gains" and strengthens the VN claim. If the gain shrinks, we report that part of the VN gain reflects robustness to noisy samples, overlapping with the role of Huber — an equally honest and useful conclusion. We write whichever the data show.

**Location in revised manuscript.** Section 7.1 (Huber subsection: the evaluation-lens result plus the strict-training S14 replication); SI Table S10 ("Effect of label-noise filtering," full clean/noisy metrics for every model) and the strict-subset SI section (S14 strict-training paragraph).

**New evidence.** Table S10 (clean-subset vs noisy-complement metrics, both sources, including the ALIGNN baseline); S14 strict-training metrics completing the triad (MAE 0.1936 vs 0.2001 for the MSE counterpart).

**Progress.** Strict-subset re-scoring of existing predictions `[DONE]`. Strict-clean Huber retraining (batch P1) `[DONE]` — the Huber gain persists under a fully strict training and evaluation lens.

---

## Comment 7 — "HZW" undefined; internal series codenames

> *"HZW" is never defined or cited, and the T/E/S series labels read as internal lab notation.*

**Response.** We accept this. HZW refers to a proprietary band-gap dataset provided by **Hongzhiwei Technology (Shanghai) Co., Ltd.** (鸿之微科技（上海）股份有限公司, hzwtech.com); each record cross-references a Materials Project entry. The revised manuscript defines HZW with this full name at first use and adds a Data-availability statement noting "used with permission; access subject to the provider's terms." One detail-level footnote — the exact DFT code and functional used by the provider — will be completed from the provider's documentation; it does not block the disclosure. The level of theory for the MP labels is now stated explicitly in Section 3 (Kohn–Sham gaps at the GGA-PBE level, with GGA+U corrections for certain transition-metal compounds, citing Jain et al. 2013). We also rename the series throughout: models are indexed T1–T10 for reference, but the main text and figures use descriptive names (CGCNN / ALIGNN / M3GNet; ALIGNN+VN; ALIGNN+VN+Huber; ALIGNN+VN+Huber+decay), with the E and S lineages defined once as "exploratory HZW-stage models" and "strict-filter stage models," respectively, appearing mainly in the SI. In-figure labels are regenerated from the plotting scripts (not hand-edited).

**Location in revised manuscript.** Abstract and Keywords (HZW full name at first use); Section 3 (data-source description and citation/footnote); throughout (series renaming); all figures with series labels (regenerated); Data availability.

**New evidence.** SI Table S6 (T-series ↔ descriptive-name mapping); otherwise a terminology and citation change.

**Progress.** Series renaming and figure regeneration `[DONE]`. HZW company-name disclosure `[DONE]` (confirmed by advisor: Hongzhiwei Technology (Shanghai) Co., Ltd.). DFT code/functional footnote to be copied from provider documentation — non-blocking.

---

## Comment 8 — Metal / non-metal F1 threshold undefined

> *Table 3 reports F1 for metal/non-metal classification without stating the threshold used.*

**Response.** We accept this. In our implementation, a ground-truth metal is a band gap ≤ 1×10⁻⁸ eV (database-reported zero gap), and a predicted metal is a predicted value ≤ 0.1 eV; the asymmetry exists because a regression output is essentially never exactly zero and needs a tolerance. We add an explicit definition and, because three distinct metal-related thresholds appear across the project (audit conflict statistics; T-series data filtering; F1 evaluation), we present all three in one clearly labeled table so they are not confused. We also note that because zero/non-zero conflicts were removed during filtering, the final MP and HZW metal ground-truth labels are identical — which explains why the CGCNN F1 is nearly identical across the two sources. To show the ranking is not an artifact of the threshold choice, we add a small sensitivity table over predicted-metal thresholds {0.01, 0.05, 0.1}. Across these three thresholds and both label sources (n = 3,803 ground-truth metals; the table now also includes the no-VN ALIGNN baseline), the ALIGNN baseline is uniformly the lowest F1 at every threshold on both sources, the plain virtual-node model is next, and every virtual-node variant exceeds the baseline at every threshold; the top rank still alternates between the Huber and decay variants within about 0.004 F1 — a near-tie whose leader depends on source and threshold — confirming the ordering among the leading variants is not an artifact of the threshold.

**Location in revised manuscript.** Section 5 (F1 protocol definition); Section 3.2 (threshold-inventory table / footnote); SI (threshold sensitivity table).

**New evidence.** Threshold-inventory table; Table S11 (F1 threshold sensitivity, including the ALIGNN baseline).

**Progress.** Definition and inventory table `[DONE]`. F1 sensitivity table `[DONE]`.

---

## Comment 9 — Missing classic citations for oversmoothing and virtual nodes

> *The introduction and related work omit foundational graph-ML references on oversmoothing and virtual nodes / global tokens.*

**Response.** We accept this and have added the missing literature, verifying each entry before inclusion. The twelve verified additions (now in the manuscript) include, for oversmoothing, Li et al. (AAAI 2018) and Oono & Suzuki (ICLR 2020), plus Cai & Wang (2020) as the source of the Dirichlet-energy metric we use in Section 7.6 (previously used without citation); for virtual nodes / global tokens, Gilmer et al. (ICML 2017, master node), Ying et al. (Graphormer, NeurIPS 2021), and Cai et al. (ICML 2023, on the connection between virtual-node MPNNs and graph transformers, which directly supports our "VN as a cheap alternative to global attention" argument); for materials-side global modeling, Okabe et al. (Nature Computational Science 2024, virtual-node GNN for phonons) and Yan et al. (Matformer, NeurIPS 2022). We have specifically fixed the three previously citation-free passages the reviewer flagged (Introduction paragraph 3, Section 2.2, and the opening of Section 7.6 (renumbered from 7.5 after the new M3GNet discussion subsection)).

**Location in revised manuscript.** Section 1 (Introduction, paragraph 3); Section 2.2; Section 7.6; References.

**New evidence.** Not applicable (12 verified new citations, now in the manuscript).

**Progress.** `[DONE]`.

---

## Comment 10 — Hyperparameters unmotivated; no sensitivity analysis

> *The Huber δ, top-k fraction, and decay schedule appear arbitrary. Please explain their choice and provide at least a minimal sensitivity analysis.*

**Response.** We accept this. These values (δ = 0.5; top-k fraction 0.2 clipped to [4, 32]; warm-up 10 + 10; decay from epoch 10 to 50 with floor 0.25) were fixed once from prior experience and were not swept — we disclose this honestly and add sensitivity results. We (a) add a "selection rationale" statement to Sections 4.6/4.7/5 noting these are prior empirical values, that no per-variant tuning was performed, and that all variants share the same training budget (itself a fairness feature); (b) sweep Huber δ ∈ {0.25, 1.0} and the decay floor min_scale ∈ {0.5, 0.75} on both sources (the decay sweep is shared with Comment 5, run once), while explaining that top-k is not swept because it was already shown to provide no independent gain and is not a core claim; and (c) add a complete training-configuration table (optimizer, learning rate, scheduler, batch, epochs, AMP, seed, hardware, key software versions). We also correct the Huber formula in the text to match the `smooth-L1(β = 0.5)` implementation (a scaled form of the standard Huber objective).

**Location in revised manuscript.** Sections 4.6 / 4.7 / 5 (rationale statements and formula correction); SI (sensitivity table and full training-configuration table).

**New evidence.** Table S14 (training-configuration table, shared with Comment 2); Table S15 (Huber-β and decay-floor sensitivity). The one-line robust-range conclusion, now in the manuscript: the Huber gain over MSE persists for every β ∈ {0.25, 0.5, 1.0} on both sources (the spread across β — up to 0.006 eV on MP — exceeds T7's seed-level standard deviation of 0.0016 eV; the single-seed β = 0.25 probe is in fact the best MP value in the sweep, which we report honestly but do not adopt, since selecting β on test results would constitute post-hoc tuning — so we claim only that the sign of the Huber gain over MSE is robust across β), and no decay floor makes the decay variant reliably better than the no-decay model — on HZW the MAE improves monotonically towards the no-decay limit.

**Progress.** Rationale statements, formula correction, configuration table `[DONE]`. Sensitivity sweeps (8 runs, batch P1) `[DONE]` — both defaults are robust choices, not cherry-picks.

---

## 中文对照要点（供师兄快速审阅，每条一句话）

- **意见 1（多种子）**：接受；配对 bootstrap 已补全 10 对（含 T2 baseline，SI Table S7 全）——T4−T2 两源显著（MP −0.0075 CI[−0.0121,−0.0028] P=0.9995；HZW −0.0104 CI[−0.0147,−0.0060] P=1.0000），T7−T2 也两源显著（MP −0.0151、HZW −0.0181），即 VN 增益与 Huber 增益均有 bootstrap 背书，唯 decay（T10−T7）两源均不显著（MP ΔMAE −0.0012 CI[−0.0058,+0.0033] P=0.70，HZW 点估计反偏向 Huber-only +0.0030）；已诚实降级为"MAE 不可分、decay 价值在 bias 0.025→0.014 与 RMSE 0.504→0.497"，主推 T7；3 seeds 均值±std 已全部完成（6 核心模型 × 2 标签源 × 3 seeds 42/1042/2042 = 24 runs，修订 Table 3 改三种子 mean±std、逐种子明细见 SI "Multi-seed replication"）——方向一致性两源全 3/3：T4−T2 MP −0.0080±0.0006、HZW −0.0104±0.0011；T7−T4 MP −0.0088±0.0013、HZW −0.0086±0.0009；decay(T10−T7) 小且符号不稳（MP −0.0004±0.0017 2/3 同号、HZW +0.0018±0.0023 2/3 同号），种子复现与配对 bootstrap 独立得出同一结论、T10 按预注册降级；两个新 seed 用 val-MAE patience=8 早停（仅 M3GNet 触发、epoch 49 停 best 41、不影响 best-val 口径），测试集样本 id 跨 seed 逐字节一致已验证，均写入修订稿第 5 节。
- **意见 2（M3GNet 弱）**：承认从头训练且未调参；三个单因子调参探针已跑完（epochs 100 → 0.2867、lr 5e-4 → 0.2912、units 256 → 0.3112，基线种子带 0.2897±0.0033，T2 为 0.2400）——没有一个能缩小差距，写实结论"单因子探针下不像欠调参（单种子探针，联合搜索超出范围）"（SI Table S16），主线不变。
- **意见 3（240 样本）**：接受；全测试集 8,455×2 源重跑已完成并全面替换——>80 原子子集（n=422）plain-VN 与 Huber 变体三指标 CI 两源全部不跨 0（240 样本时 Huber 的 HZW Dirichlet 还跨 0，全量后不再跨），decay 变体仅剩 HZW Dirichlet 一处跨 0；信号保留且更强，按预注册升级为全量证据；Table 4 / 7.6 节 / Fig 4(c,d) / SI Table S12 已全部改为全量口径，240 样本作为代表性交叉检查保留于 SI 文字。
- **意见 4（第二性质）**：按师兄决策不做形成能实验——HZW 无任何第二性质标签，做形成能必须放弃跨源设计、与论文主轴矛盾；论文收紧为纯带隙研究，以 Limitations 首条范围声明+Okabe et al. (VGNN) 文献回应，size-conditional 诊断定位为范围内机制证据；另新增 element/symmetry-resolved 误差分析（新小节+新 Figure 5：per-element MAE 周期表热图、T7−T2 元素级改善热图、按晶系 MAE、按带隙分桶 MAE），在带隙目标内部提供物理分辨证据、部分回应"物理洞见"关切（已完成，无需 GPU/API）。
- **意见 5（decay 机制）**：接受，性价比最高；四类零成本证据已完成替换猜测——训练动态（HZW 深水区 gap +0.005~0.008 eV 且后段爬升）、交叉评估（符号由训练源决定，非评估标签假象）、群体画像（过渡金属+4.8pp 等 4 维显著、分离度弱如实说）、全局广播分解（MP 降 bias/RMSE、HZW 反升）；7.2 已重写为三条数据证据；剂量-响应扫描也已完成——HZW MAE 随 decay 减弱单调恢复（0.1970→0.1950→0.1943→无 decay 0.1940，SI Table S15），机制链闭合。
- **意见 6（strict 子集）**：接受；测试子集重评已完成——clean(7958) Huber 增益两源保留（MP −0.0089、HZW −0.0071），noisy(497) 高方差且源相关（MP 翻正 +0.0140、HZW 增大 −0.0186）；T4−T2 架构增益对照直接回答"是否被掩盖"：MP clean −0.0081 / noisy +0.0015（增益确被噪声掩盖），HZW clean −0.0101 / noisy −0.0141（方向相反，掩盖为源相关）；strict-Huber 训练（S14）也已跑完：strict 全口径 MAE 0.2001(S4/MSE)→0.1936(S14/Huber)，增益 −0.0064 与 relaxed 口径同量级，Huber 改进在按构造去噪的数据上依然成立，三元组闭合。
- **意见 7（HZW/代号）**：接受；HZW = Hongzhiwei Technology (Shanghai) Co., Ltd.（鸿之微科技（上海）股份有限公司，hzwtech.com），名称已由师兄确认定稿并入文；系列改描述名、图重生成（已完成）；仅 DFT 代码/泛函小注待从 provider 文档补齐（不阻塞）。
- **意见 8（F1 阈值）**：接受；真值金属≤1e-8、预测金属≤0.1 eV，三套阈值一次讲清+敏感性表（全部已完成，表含 T2 共 24 行）；所有阈值/两源下 T2（无 VN baseline）恒最低、T4 次之、全部 VN 变体高于 baseline；T7/T10 顶位差约 0.004 的近似平局且源相关，排序非阈值假象。
- **意见 9（经典引用）**：接受；12 条 oversmoothing/虚拟节点文献已核实入文，修三处零引用段落（已完成）。
- **意见 10（超参）**：接受；诚实说明为经验值未扫参，补训练配置总表；敏感性 8 个 run 已跑完（SI Table S15）——Huber β∈{0.25,0.5,1.0} 两源全部保持对 MSE 的增益（跨 β 波动最大 0.006 eV（MP），超过 T7 种子级标准差 0.0016 eV；MP 上单种子 β=0.25 反而最优，如实报告但不采纳——按测试集选 β 属事后调参，故仅声称 Huber 对 MSE 增益的符号跨 β 稳健），decay floor {0.5,0.75} 无一让 T10 稳定超过 T7 且 HZW 单调向无 decay 恢复——两个默认值都是稳健选择而非精挑。

---

## Overall progress snapshot

| Comment | Writing / existing-data part | GPU-dependent part | External confirmation |
| --- | --- | --- | --- |
| 1 | `[DONE]` bootstrap CI (10 pairs) | `[DONE]` 24 runs (3 seeds) | — |
| 2 | `[DONE]` config + refs | `[DONE]` 3 tuning probes (no gap closure) | — |
| 3 | `[DONE]` CI on 240 | `[DONE]` full-set rerun (signal strengthened) | — |
| 4 | `[DONE]` scoping statement + literature (no experiment, per advisor decision) | — | — |
| 5 | `[DONE]` 4 zero-cost analyses | `[DONE]` dose–response (monotone HZW recovery) | — |
| 6 | `[DONE]` subset re-scoring | `[DONE]` strict-Huber S14 (0.2001 → 0.1936) | — |
| 7 | `[DONE]` renaming + figures + name disclosure (DFT footnote non-blocking) | — | — |
| 8 | `[DONE]` definition + sensitivity | — | — |
| 9 | `[DONE]` 12 citations | — | — |
| 10 | `[DONE]` rationale + config table | `[DONE]` sensitivity sweeps (defaults robust) | — |

**Remaining external dependency:** only the non-blocking DFT code/functional footnote for the HZW dataset (Comment 7), to be copied from the provider's documentation; the disclosure name itself is confirmed. All GPU batches (Comments 1/2/3/5/6/10) are complete, and no MP API key is needed since Comment 4 is addressed by the scoping statement rather than a new experiment.
