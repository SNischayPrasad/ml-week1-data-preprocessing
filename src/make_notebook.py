"""Build and execute notebooks/week1_preprocessing.ipynb.

The notebook is generated from this file rather than hand-edited, so it can never
drift from the pipeline it documents. Run it after run_all.py.
"""
import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""\
# Week 1 — Python for Machine Learning & Data Preprocessing

**Dataset:** UCI *Adult / Census Income* — 48,842 records, 14 attributes, binary income target.
**Task:** clean and preprocess the data, documenting each step.

This notebook walks through the pipeline and shows the evidence behind each decision.
The full write-up is in [`reports/REPORT.md`](../reports/REPORT.md); everything here is
reproducible with `python run_all.py`.
"""))

cells.append(code("""\
import sys, json, warnings
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from IPython.display import Image, display
from config import DATA_INTERIM, DATA_PROCESSED, OUTPUTS, FIGURES, TARGET

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

def out(name):
    return json.loads((OUTPUTS / name).read_text())

print("pandas", pd.__version__, "| numpy", np.__version__)"""))

# -- 1. loading
cells.append(md("""\
---
## 1. Loading — a label bug that would have gone unnoticed

UCI ships Adult as two files that are **not** formatted identically. `adult.test` has a junk
first line and every label carries a trailing period. Concatenating them naively gives *four*
classes instead of two, and every binary metric computed afterwards is meaningless."""))

cells.append(code("""\
a0 = out("00_assemble.json")
print("labels before the fix:", a0["labels_before_fix"])
print("labels after the fix: ", a0["labels_after_fix"])
print(f"\\nrows: {a0['adult_data_rows']:,} (adult.data) + {a0['adult_test_rows']:,} (adult.test)"
      f" = {a0['combined_rows']:,}")"""))

# -- 2. audit
cells.append(md("""\
---
## 2. Data quality audit

### 2.1 The missing values are invisible to pandas

`df.isna().sum().sum()` returns **0**. Every missing value is the literal string `?`."""))

cells.append(code("""\
df = pd.read_csv(DATA_INTERIM / "adult_raw_combined.csv", skipinitialspace=True)
a1 = out("01_audit.json")

print("NaN detected by pandas:", a1["nan_detected_by_pandas"])
print("\\nActual missingness ('?' sentinel):")
display(pd.DataFrame(a1["sentinel_missing"]).T.rename(columns={"count": "missing", "pct": "% of rows"}))
print(f"rows with at least one missing value: {a1['rows_with_any_missing']:,} "
      f"({a1['rows_with_any_missing_pct']}%)")"""))

cells.append(md("""\
### 2.2 Is the missingness random?

This question decides the imputation strategy, so it gets measured rather than assumed."""))

cells.append(code("""\
print(f"P(>50K | row is complete)        = {a1['target_rate_rows_complete']}%")
print(f"P(>50K | row has a missing value) = {a1['target_rate_rows_with_missing']}%")
print(f"\\nRatio: incomplete rows are {a1['target_rate_rows_complete']/a1['target_rate_rows_with_missing']:.2f}x "
      "less likely to be high earners.\\n")

# ...and the missingness is nested, not two independent events:
print("workclass AND occupation both missing:", f"{a1['workclass_and_occupation_both_missing']:,}")
print("workclass missing on its own:         ", a1["workclass_missing_only"])
print("occupation missing on its own:        ", a1["occupation_missing_only"])"""))

cells.append(code("""display(Image(str(FIGURES / "fig1_missingness.png")))"""))

cells.append(md("""\
**Conclusion.** The missingness is *informative* — a person with blank employment fields is
roughly half as likely to be a high earner. Mode-imputation would assert these 3,620 people are
typical private-sector workers, which the data contradicts. The absence is therefore preserved
as its own category, `"Missing"`."""))

cells.append(md("""\
### 2.3 Duplicates — including five contradictions"""))

cells.append(code("""\
feat = [c for c in df.columns if c != TARGET]
print("exact duplicate rows:              ", a1["exact_duplicate_rows"])
print("duplicates ignoring the target:    ", a1["duplicate_rows_excluding_target"])

contra = (df[df.duplicated(subset=feat, keep=False)]
          .groupby(feat, observed=True)
          .filter(lambda g: g[TARGET].nunique() > 1))
print(f"\\n{len(contra)} rows are identical in every feature but carry opposite labels:")
display(contra[["age", "workclass", "education", "occupation", "hours_per_week", TARGET]]
        .sort_values(["age", "occupation"]))"""))

cells.append(md("""\
These are **irreducible Bayes error** — real people the feature set cannot tell apart. They are
not errors to fix, and they cap the accuracy available on this dataset. The 52 *exact*
duplicates are removed, and removed **before** the split so no record can land on both sides."""))

cells.append(md("""\
### 2.4 `capital_gain` is top-coded

244 rows sit at exactly 99,999 — a census censoring code, not an amount of money."""))

cells.append(code("""\
cg = df["capital_gain"]
print(f"rows at exactly 99,999: {(cg == 99999).sum()}")
print(f"next largest value:     {sorted(cg.unique())[-2]:,}")
print(f"gap:                    {99999 - sorted(cg.unique())[-2]:,}")
print(f"share of them earning >50K: {100*(df.loc[cg == 99999, TARGET] == '>50K').mean():.0f}%")
display(Image(str(FIGURES / "fig3_capital_gain.png")))"""))

cells.append(md("""\
Handled by **flagging and clipping**, not deleting: a `capital_gain_topcoded` indicator records
the censoring, and the value is replaced with 41,310 (the largest genuine amount). Dropping the
rows would discard 244 strongly predictive records; leaving 99,999 would poison every scaler."""))

cells.append(md("""\
### 2.5 Outliers: measured, then deliberately left alone

The reflex here is to drop everything outside the 1.5×IQR fence. On this dataset that reflex
destroys a quarter of the data."""))

cells.append(code("""\
a3 = out("03_eda.json")
h = df["hours_per_week"]
q1, q3 = h.quantile(.25), h.quantile(.75)
print(f"Q1 = {q1:.0f} h, median = {h.median():.0f} h, Q3 = {q3:.0f} h  ->  IQR = {q3-q1:.0f} h")
print(f"1.5x IQR fence: [{q1-1.5*(q3-q1):.1f} h, {q3+1.5*(q3-q1):.1f} h]")
print(f"\\n{a3['fig2_pct_exactly_40']}% of the sample works exactly 40 h, which collapses Q1 onto the median.")
print(f"The rule would delete {a3['fig2_hours_flagged_n']:,} rows ({a3['fig2_hours_flagged_pct']}% of the dataset).")
display(Image(str(FIGURES / "fig2_iqr_trap.png")))"""))

cells.append(md("""\
**Decision: no automated outlier removal.** Part-time workers and 60-hour-a-week managers are
not measurement errors — they are the most informative rows in the file. The only value treated
as invalid is the 99,999 censoring code, because it was *shown* to be a code."""))

# -- 3. cleaning + split
cells.append(md("""\
---
## 3. Cleaning decisions and the split

| Decision | Evidence |
|---|---|
| Drop `fnlwgt` | a census sampling weight, not a property of the person; \\|r\\| = 0.0063 with the target |
| Drop `education`, keep `education_num` | a perfect 1-to-1 mapping; the numeric version preserves the ordering |
| `?` → `NaN` | so every downstream tool sees missingness as missingness |
| Flag + clip the top-coding | keeps "very large gain" without the fictional 58,689 gap |
| Deduplicate **before** splitting | otherwise copies of one record land on both sides |"""))

cells.append(code("""\
a2 = out("02_clean_split.json")
for k in ["rows_in", "exact_duplicates_dropped", "rows_after_dedupe", "columns_dropped",
          "sentinel_values_converted_to_nan", "capital_gain_topcoded_rows",
          "train_rows", "test_rows", "train_positive_rate_pct", "test_positive_rate_pct"]:
    print(f"{k:35s} {a2[k]}")

display(Image(str(FIGURES / "fig5_education_ordinal.png")))"""))

cells.append(md("""\
### The unseen-category trap, quantified

`native_country` has 42 levels and **`Holand-Netherlands` has exactly one row in all 48,842**.
With `random_state=42` it lands in train, so the problem is invisible — which is exactly why it
was tested across 200 seeds instead of trusted."""))

cells.append(code("""\
a2b = out("02b_unseen_category_risk.json")
print(f"seeds tested:                      {a2b['seeds_tested']}")
print(f"seeds with an unseen test category: {a2b['seeds_with_unseen_test_category']} ({a2b['pct']}%)")
print(f"levels involved:                    {a2b['levels_involved']}")
print("\\n-> the encoder uses handle_unknown='infrequent_if_exist' so unseen levels are absorbed.")"""))

# -- 4. pipeline
cells.append(md("""\
---
## 4. The preprocessing pipeline

A single `ColumnTransformer`, **fitted on train only**, applied unchanged to test.

| Group | Treatment |
|---|---|
| `capital_gain`, `capital_loss` | median impute → standardise |
| `age`, `education_num`, `hours_per_week` | median impute → standardise |
| 3 binary flags | passthrough |
| 7 categorical columns | impute `"Missing"` → one-hot, `min_frequency=1%`, unknowns absorbed |"""))

cells.append(code("""\
import joblib
from step04_pipeline import build_preprocessor

pre = joblib.load(DATA_PROCESSED / "preprocessor.joblib")
a4 = out("04_pipeline.json")
print(f"{a4['features_in']} input columns -> {a4['features_out']} features")
print(f"one-hot block: {a4['onehot_cols_naive']} columns if naive -> {a4['onehot_cols_actual']} after folding rare levels\\n")
display(pd.DataFrame(a4["categorical_encoding"]).T)"""))

cells.append(md("""\
### Proof that no leakage occurred

After scaling, the **training** features have mean exactly 0. The **test** features do not — and
that is the point. A test mean of exactly 0 would mean the scaler had seen the test split."""))

cells.append(code("""\
display(pd.DataFrame({
    "train mean after scaling": a4["train_mean_after_scaling"],
    "test mean after scaling": a4["test_mean_after_scaling"],
}))
print("Non-zero test means = the scaler never saw the test split.")"""))

# -- 5. feature selection
cells.append(md("""\
---
## 5. Feature selection — three views

Mutual information (non-linear dependence), feature-feature correlation (redundancy), and
permutation importance (what a model actually uses). A feature is only dropped when all three
agree."""))

cells.append(code("""\
a5 = out("05_feature_selection.json")
display(Image(str(FIGURES / "fig6_mutual_information.png")))

print("Redundancies the correlation view exposes:")
display(pd.DataFrame(a5["top_correlated_pairs"]).head(4))"""))

cells.append(code("""\
print("features condemned by all three views:", a5["agreed_dead_features"])
print(f"test AUC with all {a5['n_features']} features: {a5['auc_all_features']}")
print(f"test AUC after dropping them:         {a5['auc_after_dropping_dead']}")
print(f"cost of dropping them:                {a5['auc_delta']:+.5f}\\n")

curve = pd.DataFrame(a5["topk_curve"])
curve["% of full AUC"] = (100 * curve["auc"] / a5["auc_all_features"]).round(2)
display(curve)
print("10 features (19% of them) recover 99% of the full-model AUC.")"""))

# -- 6. ablations
cells.append(md("""\
---
## 6. What each decision was actually worth

Each knob flipped with everything else held fixed. The grey band is ±1 standard error of the
cross-validated mean — anything inside it is not a finding."""))

cells.append(code("""\
display(Image(str(FIGURES / "fig7_effect_sizes.png")))

e = out("06_experiments.json")
c = e["C_scaling"]
print("Scaling is model-dependent:")
for m, name in [("knn", "KNN (distance-based)"), ("logreg", "Logistic regression"),
                ("hgb", "Gradient boosting")]:
    print(f"  {name:28s} {c[m]['auc_gain_from_scaling']:+.5f} AUC")
print(f"\\nAnd for logistic regression it is really about convergence:"
      f"\\n  fit time scaled = {c['logreg']['scaled']['seconds']}s"
      f"   unscaled = {c['logreg']['unscaled']['seconds']}s")"""))

cells.append(md("""\
### Where the first draft was wrong

`capital_gain` has skew 7.637 and is 91.7% zeros, so the first version of this pipeline applied
`log1p`. Measuring it showed the transform **costs** performance on every model that can notice
it — and is *exactly* zero for trees, as theory predicts for a monotone transform."""))

cells.append(code("""\
d = e["D_log1p"]
log_by_model = out("06b_log1p_by_model.json")
rows = []
for m, label in [("logreg", "Logistic regression"), ("knn", "KNN"), ("hgb", "Gradient boosting")]:
    rows.append({"model": label,
                 "with log1p": log_by_model[f"{m}_log1p"],
                 "without log1p": log_by_model[f"{m}_raw"],
                 "delta": round(log_by_model[f"{m}_log1p"] - log_by_model[f"{m}_raw"], 5)})
display(pd.DataFrame(rows))
print("log1p was removed. It stays behind a USE_LOG1P switch so the claim remains falsifiable.")"""))

cells.append(md("""\
### Leakage — the same mistake, two orders of magnitude apart"""))

cells.append(code("""\
L = e["E_leakage"]
e1, e2 = L["E1_scaler_fitted_on_train_plus_test"], L["E2_target_encoding_before_split"]
print(f"StandardScaler fitted on train+test:       {e1['optimism']:+.5f} AUC of phantom gain")
print(f"Target encoding ({e2['key_cardinality']:,} keys) before split: {e2['optimism']:+.5f} AUC of phantom gain")
print(f"   (test AUC {e2['auc_correct']} done correctly -> {e2['auc_leaky']} with the leak)")
display(Image(str(FIGURES / "fig8_leakage.png")))"""))

cells.append(md("""\
**The transferable lesson:** leakage severity scales with how much the transform can memorise
individual rows. A two-parameter scaler can memorise nothing; a 4,233-level encoding can
memorise almost everything."""))

# -- 7. results
cells.append(md("""\
---
## 7. Final results

Classes are imbalanced 3.18:1, so accuracy is reported but never optimised — predicting
`<=50K` for everybody already scores 76.06%."""))

cells.append(code("""\
f = e["F_final"]
res = pd.DataFrame([
    {"model": "Majority class", "CV AUC": None, "test AUC": 0.5, "test PR-AUC": None,
     "test F1 (>50K)": 0.0, "test accuracy": e["baseline_majority"]["test_accuracy"]},
    {"model": "Logistic regression", "CV AUC": f["logreg"]["cv_auc_train"],
     "test AUC": f["logreg"]["test_auc"], "test PR-AUC": f["logreg"]["test_pr_auc"],
     "test F1 (>50K)": f["logreg"]["test_f1_positive"], "test accuracy": f["logreg"]["test_accuracy"]},
    {"model": "Gradient boosting", "CV AUC": f["hist_gradient_boosting"]["cv_auc_train"],
     "test AUC": f["hist_gradient_boosting"]["test_auc"],
     "test PR-AUC": f["hist_gradient_boosting"]["test_pr_auc"],
     "test F1 (>50K)": f["hist_gradient_boosting"]["test_f1_positive"],
     "test accuracy": f["hist_gradient_boosting"]["test_accuracy"]},
]).set_index("model")
display(res)
display(Image(str(FIGURES / "fig9_final_curves.png")))"""))

cells.append(md("""\
---
## 8. Limitations

1. The `"Missing"`-category result (+0.00160 AUC) sits just **below** the noise floor; it is
   defended on the MNAR evidence, not on that number.
2. Top-coding was handled by clipping. A censored-regression (Tobit-style) treatment would be
   more principled.
3. **Fairness was not assessed.** The model uses `race` and `sex`; a deployed system would need
   subgroup performance and a decision about whether those attributes belong in it at all.
4. Headline numbers come from a single split; repeated splits would tighten the intervals on the
   smaller effects.

Full discussion in [`reports/REPORT.md`](../reports/REPORT.md)."""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

path = NB_DIR / "week1_preprocessing.ipynb"
print(f"executing {len(cells)} cells...")
NotebookClient(nb, timeout=900, kernel_name="python3",
               resources={"metadata": {"path": str(NB_DIR)}}).execute()
nbf.write(nb, path)

n_err = sum(1 for c in nb.cells
            if any(o.get("output_type") == "error" for o in c.get("outputs", [])))
print(f"wrote {path}  ({len(cells)} cells, {n_err} errors)")
raise SystemExit(1 if n_err else 0)
