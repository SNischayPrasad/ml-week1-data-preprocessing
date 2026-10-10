"""Cross-check every number quoted in reports/REPORT.md against outputs/*.json.

Run after any change to the pipeline. A FAIL means the report and the code have
drifted apart.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPORT = (ROOT / "reports" / "REPORT.md").read_text(encoding="utf-8")


def load(name):
    return json.loads((OUT / name).read_text())


A0, A1, A2 = load("00_assemble.json"), load("01_audit.json"), load("02_clean_split.json")
A2B = load("02b_unseen_category_risk.json")
A3, A4 = load("03_eda.json"), load("04_pipeline.json")
A5, A6, A7 = load("05_feature_selection.json"), load("06_experiments.json"), load("07_summary.json")

checks = []


def check(label, claim_in_report, actual, fmt="{}"):
    """Assert the literal string appears in the report AND matches the data."""
    txt = fmt.format(actual)
    present = claim_in_report in REPORT
    matches = claim_in_report == txt
    checks.append((label, claim_in_report, txt, present and matches))


# --- §1 headline ------------------------------------------------------------
check("raw rows", "48,842", A0["combined_rows"], "{:,}")
check("train file rows", "32,561", A0["adult_data_rows"], "{:,}")
check("test file rows", "16,281", A0["adult_test_rows"], "{:,}")
check("rows after dedupe", "48,790", A2["rows_after_dedupe"], "{:,}")
check("duplicates dropped", "52", A2["exact_duplicates_dropped"], "{}")
check("sentinel total raw", "6,465",
      sum(v["count"] for v in A1["sentinel_missing"].values()), "{:,}")
check("sentinel post-dedupe", "6,456", A2["sentinel_values_converted_to_nan"], "{:,}")
check("features out", "53", A4["features_out"], "{}")
check("train rows", "39,032", A2["train_rows"], "{:,}")
check("test rows", "9,758", A2["test_rows"], "{:,}")
check("train pos rate", "23.942", A2["train_positive_rate_pct"], "{}")
check("test pos rate", "23.939", A2["test_positive_rate_pct"], "{}")
check("majority acc", "76.061", 100 * A6["baseline_majority"]["test_accuracy"], "{:.3f}")

# --- final models -----------------------------------------------------------
lr, hgb = A6["F_final"]["logreg"], A6["F_final"]["hist_gradient_boosting"]
check("logreg auc", "0.91228", lr["test_auc"], "{:.5f}")
check("logreg prauc", "0.78276", lr["test_pr_auc"], "{:.5f}")
check("logreg f1", "0.67699", lr["test_f1_positive"], "{:.5f}")
check("logreg acc", "0.85889", lr["test_accuracy"], "{:.5f}")
check("hgb auc", "0.93239", hgb["test_auc"], "{:.5f}")
check("hgb prauc", "0.84022", hgb["test_pr_auc"], "{:.5f}")
check("hgb f1", "0.72295", hgb["test_f1_positive"], "{:.5f}")
check("hgb acc", "0.87692", hgb["test_accuracy"], "{:.5f}")
check("logreg cv", "0.90804", lr["cv_auc_train"], "{:.5f}")
check("logreg cv std", "0.00479", lr["cv_auc_std"], "{:.5f}")
check("hgb cv", "0.92759", hgb["cv_auc_train"], "{:.5f}")
check("hgb cv std", "0.00413", hgb["cv_auc_std"], "{:.5f}")
check("standard error", "0.00214", A7["standard_error_cv_mean"], "{:.5f}")

# --- §4 audit ---------------------------------------------------------------
sm = A1["sentinel_missing"]
check("occupation missing", "2,809", sm["occupation"]["count"], "{:,}")
check("workclass missing", "2,799", sm["workclass"]["count"], "{:,}")
check("native_country missing", "857", sm["native_country"]["count"], "{:,}")
check("rows any missing", "3,620", A1["rows_with_any_missing"], "{:,}")
check("rows any missing pct", "7.41", A1["rows_with_any_missing_pct"], "{:.2f}")
check("rate incomplete", "13.23", A1["target_rate_rows_with_missing"], "{:.2f}")
check("rate complete", "24.78", A1["target_rate_rows_complete"], "{:.2f}")
check("workclass only", "0", A1["workclass_missing_only"], "{}")
check("occupation only", "10", A1["occupation_missing_only"], "{}")
check("dup excl target", "57", A1["duplicate_rows_excluding_target"], "{}")
check("imbalance ratio", "3.179", A1["target_imbalance_ratio"], "{}")
check("majority baseline pct", "76.07", A1["majority_class_baseline_acc"], "{:.2f}")
check("cg pct zero", "91.74", A1["numeric"]["capital_gain"]["pct_zero"], "{:.2f}")
check("cg skew raw", "11.895", A1["numeric"]["capital_gain"]["skew"], "{}")
check("cl pct zero", "95.3", A1["numeric"]["capital_loss"]["pct_zero"], "{:.1f}")
check("raw scale ratio", "41,091", A1["scale_spread_max_over_min_std"], "{:,.0f}")
check("fnlwgt corr", "0.0063", abs(A1["fnlwgt_corr_with_target"]), "{:.4f}")
check("max spearman", "0.1644", A1["max_abs_offdiag_numeric_corr"], "{:.4f}")
check("edu_num outliers", "1,794", A1["numeric"]["education_num"]["n_outliers_iqr"], "{:,}")
check("edu_num outlier pct", "3.67", A1["numeric"]["education_num"]["pct_outliers_iqr"], "{:.2f}")

# --- §5 / EDA ---------------------------------------------------------------
check("hours flagged n", "13,496", A3["fig2_hours_flagged_n"], "{:,}")
check("hours flagged pct", "27.63", A3["fig2_hours_flagged_pct"], "{:.2f}")
check("pct exactly 40", "46.69", A3["fig2_pct_exactly_40"], "{:.2f}")
check("edu rate at 16", "71.05", A3["fig5_rate_at_16"], "{}")
check("trimmed scale ratio", "1,479", A3["fig4_std_ratio"], "{:,.0f}")
check("topcoded rows", "244", A2["capital_gain_topcoded_rows"], "{}")
check("true max", "41,310", A2["capital_gain_true_max_observed"], "{:,}")
check("cg skew after clip", "7.637", A4["skew_before_log1p"]["capital_gain"], "{}")
check("contradictory rows", "10", A2["contradictory_label_rows"], "{}")

# --- §6 split / leakage discipline -----------------------------------------
check("unseen seeds", "45", A2B["seeds_with_unseen_test_category"], "{}")
check("unseen pct", "22.5", A2B["pct"], "{}")
check("test mean age", "+0.019402", A4["test_mean_after_scaling"]["age"], "{:+.6f}")
check("test mean hours", "−0.025977",
      str(A4["test_mean_after_scaling"]["hours_per_week"]).replace("-", "−"), "{}")
check("test mean edu", "+0.000698", A4["test_mean_after_scaling"]["education_num"], "{:+.6f}")

# --- §7 pipeline ------------------------------------------------------------
check("onehot naive", "86", A4["onehot_cols_naive"], "{}")
check("onehot actual", "45", A4["onehot_cols_actual"], "{}")
nc = A4["categorical_encoding"]["native_country"]
check("nc folded", "39", nc["levels_folded_as_infrequent"], "{}")
check("nc output cols", "4", nc["output_columns"], "{}")

# --- §8 ablations -----------------------------------------------------------
E = A6
knn_gain = E["C_scaling"]["knn"]["auc_gain_from_scaling"]
check("knn scaling gain", "+0.01878", knn_gain, "{:+.5f}")
check("logreg scaling gain", "+0.00036", E["C_scaling"]["logreg"]["auc_gain_from_scaling"], "{:+.5f}")
check("hgb scaling gain", "+0.00000", E["C_scaling"]["hgb"]["auc_gain_from_scaling"], "{:+.5f}")
d_log = E["D_log1p"]["without_log1p"]["test_auc"] - E["D_log1p"]["with_log1p"]["test_auc"]
check("log1p delta", "+0.00241", d_log, "{:+.5f}")
d_mis = (E["A_missing"]["explicit_Missing_category"]["test_auc"]
         - E["A_missing"]["most_frequent_imputation"]["test_auc"])
check("missing delta", "+0.00160", d_mis, "{:+.5f}")
d_rare = (E["B_rare_levels"]["min_frequency_1pct"]["test_auc"]
          - E["B_rare_levels"]["naive_one_hot"]["test_auc"])
check("rare delta", "−0.00012", f"{d_rare:+.5f}".replace("-", "−"), "{}")
check("naive feature count", "94", E["B_rare_levels"]["naive_one_hot"]["n_features"], "{}")
# Wall-clock timings are machine-dependent, so assert the magnitude, not the value.
_t = E["C_scaling"]["logreg"]
_ratio = _t["unscaled"]["seconds"] / _t["scaled"]["seconds"]
checks.append(("scaling speedup >5x", f"{_ratio:.1f}x", ">5x", _ratio > 5))
L = E["E_leakage"]
check("E1 optimism", "−0.00002",
      f"{L['E1_scaler_fitted_on_train_plus_test']['optimism']:+.5f}".replace("-", "−"), "{}")
check("E2 optimism", "+0.02587", L["E2_target_encoding_before_split"]["optimism"], "{:+.5f}")
check("E2 cardinality", "4,233", L["E2_target_encoding_before_split"]["key_cardinality"], "{:,}")
check("E2 leaky", "0.79052", L["E2_target_encoding_before_split"]["auc_leaky"], "{:.5f}")
check("E2 correct", "0.76465", L["E2_target_encoding_before_split"]["auc_correct"], "{:.5f}")
check("log1p logreg with", "0.90987", E["D_log1p"]["with_log1p"]["test_auc"], "{:.5f}")

# --- §10 feature selection --------------------------------------------------
check("mi married", "0.1046", A5["mi_top10"]["marital_status_Married-civ-spouse"], "{:.4f}")
check("mi husband", "0.0906", A5["mi_top10"]["relationship_Husband"], "{:.4f}")
check("mi capital_gain", "0.0827", A5["mi_top10"]["capital_gain"], "{:.4f}")
check("mi age", "0.0710", A5["mi_top10"]["age"], "{:.4f}")
check("mi education", "0.0616", A5["mi_top10"]["education_num"], "{:.4f}")
check("perm nonpositive", "14", A5["perm_non_positive_count"], "{}")
check("auc all features", "0.91228", A5["auc_all_features"], "{:.5f}")
check("n agreed dead", "2", A5["n_agreed_dead"], "{}")
for c in A5["topk_curve"]:
    check(f"topk k={c['k']}", f"{c['auc']:.5f}", c["auc"], "{:.5f}")

# --- report ----------------------------------------------------------------
fails = [c for c in checks if not c[3]]
print(f"{'CLAIM':<28} {'IN REPORT':>14} {'FROM DATA':>14}   OK")
print("-" * 66)
for label, claimed, actual, ok in checks:
    if not ok:
        print(f"{label:<28} {claimed:>14} {actual:>14}   FAIL")
print("-" * 66)
print(f"{len(checks) - len(fails)}/{len(checks)} checks passed")
sys.exit(1 if fails else 0)
