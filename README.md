# Robust Global Virtual-Node GNNs for Crystal Band-Gap Prediction

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
  Redistributable under the Materials Project terms.
- **HZW** — a proprietary DFT band-gap database from Hongzhiwei Technology
  (Shanghai) Co., Ltd., used with permission. **These labels are not
  redistributable and are excluded from this repository**, along with all
  derived tables under `results/` that embed them.

The MP identifier list plus the audit and export scripts here are sufficient to
reproduce the MP-label results and the structure-first MP–HZW audit.

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
