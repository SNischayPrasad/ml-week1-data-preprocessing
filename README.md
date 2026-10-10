# Week 1 — Python for Machine Learning & Data Preprocessing

Clean and preprocess a sample dataset, documenting every preprocessing step and the reasoning
behind it.

**Dataset:** UCI *Adult / Census Income* — 48,842 records, 14 attributes, binary income target.
**Write-up:** [`reports/REPORT.md`](reports/REPORT.md) ← start here (also as [Word](reports/Week1_Preprocessing_Report.docx), [PDF](reports/Week1_Preprocessing_Report.pdf) and [HTML](reports/Week1_Preprocessing_Report.html)).
**Notebook:** [`notebooks/week1_preprocessing.ipynb`](notebooks/week1_preprocessing.ipynb)

## Quick start

```bash
pip install -r requirements.txt
python run_all.py
```

This downloads the raw data from UCI and regenerates every number, table and figure in the
report — roughly 5 minutes on a laptop CPU. To confirm the write-up still matches the code:

```bash
python src/verify_report.py
```

To regenerate the Word/PDF/HTML versions of the report:

```bash
python src/make_pdf.py      # HTML + PDF
npm install docx && node src/make_docx.js   # Word
```

It re-reads `reports/REPORT.md`, cross-checks 93 quoted figures against `outputs/*.json`, and
exits non-zero on any mismatch.

## What was done

| Step | Script | Output |
|---|---|---|
| 0 | `step00_assemble.py` | Load both UCI files; fix a label bug that produced 4 classes instead of 2 |
| 1 | `step01_audit.py` | Data quality audit — missingness, duplicates, skew, cardinality, imbalance |
| 2 | `step02_clean_split.py` | Cleaning decisions, then a stratified 80/20 split |
| 2b | `step02b_unseen_category_risk.py` | Across 200 seeds, 22.5% of splits hand the encoder an unseen category |
| 3 | `step03_eda.py` | Figures 1–5 |
| 4 | `step04_pipeline.py` | The preprocessing pipeline, fitted on train only |
| 5 | `step05_feature_selection.py` | Mutual information, correlation, permutation importance |
| 6 | `step06_experiments.py` | Ablations: what each decision was worth, plus two leakage experiments |
| 7 | `step07_summary_figures.py` | Figures 7–9 |

## Headline numbers

- **6,465** missing values found, all hidden behind the string `?` — `pandas.isna()` reported **0**
- **48,842 → 48,790** rows after removing 52 exact duplicates (5 pairs carry contradictory labels)
- **15 → 53** features; the one-hot block was cut from 86 columns to 45 by folding rare levels
- Missingness is **not random**: incomplete rows earn >50K 13.23% of the time vs 24.78% for complete rows
- Final test ROC-AUC **0.93239** (gradient boosting) against a 76.061% majority-class baseline
- Fitting a target encoding **before** the split inflated test AUC by **+0.02587**; the same error
  on a `StandardScaler` was worth **−0.00002**

Two decisions in the first draft turned out to be wrong when measured, and the report says so
rather than hiding it — see §9 of the report.

## Layout

```
run_all.py            reproduce everything end to end
requirements.txt      pinned versions
src/                  config, shared plotting style, the eight numbered steps, verify_report.py
data/raw/             untouched UCI downloads
data/interim/         combined, label-corrected
data/processed/       train/test splits (clean and encoded) + preprocessor.joblib
outputs/              every number in the report, as JSON/CSV
reports/REPORT.md     the write-up
reports/figures/      fig1 – fig9
```

Trained artefacts are regenerable; `data/` and `outputs/` can be deleted and rebuilt with
`python run_all.py`.
