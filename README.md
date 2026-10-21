# Robust Global Virtual-Node GNNs for Crystal Band-Gap Prediction

This repository is the project code repository cited in the paper's Data and
code availability statement.

Code, configurations, and analysis scripts for a study of global virtual-node
mechanisms on an ALIGNN backbone for crystal band-gap regression, evaluated
under two independent DFT label sources on an identical structure split.

Headline result: a single *plain* virtual node plus a Huber objective is the
most robust design. More elaborate global mechanisms (attentive tokens, dual
slots, top-`k` aggregation, broadcast decay) do not transfer across label
sources or across training seeds.

## Layout

| Path | Contents |
| --- | --- |
| `crystal_gnn/` | Data pipeline, models, utilities |
| `faithful_gnn/` | Reference-faithful CGCNN / ALIGNN / M3GNet runners and the ALIGNN virtual-node variants |
| `configs/` | One YAML per experiment (E-, S-, and T-series) |
| `scripts/` | Analysis: paired bootstrap, multi-seed aggregation, cross-label evaluation, over-smoothing hooks, figure builders |
| `tests/` | Unit tests for the variant implementations and dataset export |
| `docs/` | Design notes and dated implementation plans |
| `figures/` | Figure sources and rendered panels used by the paper |
| `*.tex` | Paper, supplementary, and slide sources |

## Data

Two label sources over the same structures:

- **Materials Project** — GGA-PBE (with GGA+U where applicable) band gaps.
  The benchmark labels were compiled from the public Materials Project
  database via its summary API
  (`api.materialsproject.org/materials/summary/`), queried by material id
  with a fixed field set. Because the database is continuously updated, the
  labels correspond to a snapshot retrieved between 10 April and 24 May 2026
  rather than a fixed release version. The analysis code in this repository
  performs the MP-side preprocessing and can re-fetch labels for any id set
  from the public API.
- **HZW** — a proprietary DFT band-gap database from Hongzhiwei Technology
  (Shanghai) Co., Ltd., used with permission. **These labels are not
  redistributable and are excluded from this repository**, along with all
  derived tables under `results/` that embed them.

The analysis code in this repository supports reproduction of the MP-label
experiments and MP-side preprocessing once the benchmark id list is
supplied; the id list derives from the HZW cross-reference field and is not
released here. Reproducing the exact benchmark population, the MP--HZW
structure matching, and the cross-source label-consistency audit requires
authorized access to the proprietary HZW records, subject to the provider's
terms.

## Setup

```bash
pip install -r requirements.txt
```

Training and export scripts read a Materials Project API key from
`~/.mp_api_key`; no credentials are stored in this repository.

## Building the paper

```bash
pdflatex gnn_bandgap_research_paper_draft_en.tex   # run twice to resolve refs
```
