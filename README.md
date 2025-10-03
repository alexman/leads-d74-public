# LeADS D7.4 (Public) — Algorithm for Data Mining for Legal Analysis & Policymaking

Public Phase-1 components for EU H2020 **LeADS** (GA 956562) deliverable **D7.4**.

**Phased release strategy**
- **Phase 1 (MIT)** — this repo: safe preprocessing utilities (data explosion public variant, temporal parsing), minimal validation, basic visualization, and demo notebooks.
- **Phase 2 (EUPL/GPL, after publication)** — full ML/DL pipelines (temporal normalization, learned encoders, clustering & severity prediction; ~82% accuracy in internal evaluation), schema contracts, reporting templates.

## What’s here
- `preprocessing/` — public “data explosion” (aligned multi-value split), time parsing, ID normalization, basic quality report
- `visualization/` — matplotlib-only EDA helpers
- `notebooks/` — quick demos on a small sample CSV
- `docs/` — implementation guide & data requirements

## Quickstart
```bash
# from repo root
pip install -r requirements.txt

# optional quick smoke test (saves PNGs in plots/)
python quick_test.py
