"""Step 1 - Data quality audit. Every claim in the report traces back to here.

This does not clean anything. It measures, so that each cleaning decision taken
later can be justified with a number rather than a habit.
"""
import json
import numpy as np
import pandas as pd
from config import (CATEGORICAL_COLS, DATA_INTERIM, NUMERIC_COLS, OUTPUTS,
                    SENTINEL_MISSING, TARGET)

pd.set_option("display.width", 200)

def main() -> None:
    df = pd.read_csv(DATA_INTERIM / "adult_raw_combined.csv", skipinitialspace=True)
    audit = {"n_rows": int(len(df)), "n_cols": int(df.shape[1])}

    # --- 1. Missingness: pandas sees zero NaN; the real missingness is '?' ----
    audit["nan_detected_by_pandas"] = int(df.isna().sum().sum())
    sentinel_counts = {}
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            n = int(df[c].isin(SENTINEL_MISSING).sum())
            if n:
                sentinel_counts[c] = {"count": n, "pct": round(100 * n / len(df), 3)}
    audit["sentinel_missing"] = sentinel_counts
    mask_any = df[list(sentinel_counts)].isin(SENTINEL_MISSING).any(axis=1)
    audit["rows_with_any_missing"] = int(mask_any.sum())
    audit["rows_with_any_missing_pct"] = round(100 * mask_any.mean(), 3)

    # Is missingness random, or does it carry signal? (MCAR sanity check)
    audit["target_rate_overall"] = round(100 * (df[TARGET] == ">50K").mean(), 3)
    audit["target_rate_rows_with_missing"] = round(
        100 * (df.loc[mask_any, TARGET] == ">50K").mean(), 3)
    audit["target_rate_rows_complete"] = round(
        100 * (df.loc[~mask_any, TARGET] == ">50K").mean(), 3)

    # Do workclass and occupation go missing together?
    wc = df["workclass"].isin(SENTINEL_MISSING)
    oc = df["occupation"].isin(SENTINEL_MISSING)
    audit["workclass_and_occupation_both_missing"] = int((wc & oc).sum())
    audit["workclass_missing_only"] = int((wc & ~oc).sum())
    audit["occupation_missing_only"] = int((~wc & oc).sum())

    # --- 2. Duplicates ------------------------------------------------------
    audit["exact_duplicate_rows"] = int(df.duplicated().sum())
    audit["duplicate_rows_excluding_target"] = int(
        df.drop(columns=[TARGET]).duplicated().sum())

    # --- 3. Target balance --------------------------------------------------
    vc = df[TARGET].value_counts()
    audit["target_counts"] = {str(k): int(v) for k, v in vc.items()}
    audit["target_imbalance_ratio"] = round(float(vc.max() / vc.min()), 3)
    audit["majority_class_baseline_acc"] = round(100 * float(vc.max() / vc.sum()), 3)

    # --- 4. Numeric distributions, skew, outliers ---------------------------
    num = {}
    for c in NUMERIC_COLS:
        s = df[c].astype(float)
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        num[c] = {
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "q1": round(float(q1), 2),
            "median": round(float(s.median()), 2),
            "q3": round(float(q3), 2),
            "max": round(float(s.max()), 2),
            "skew": round(float(s.skew()), 3),
            "pct_zero": round(100 * float((s == 0).mean()), 3),
            "n_outliers_iqr": int(((s < lo) | (s > hi)).sum()),
            "pct_outliers_iqr": round(100 * float(((s < lo) | (s > hi)).mean()), 3),
            "range_ratio_to_age": None,
        }
    audit["numeric"] = num
    audit["scale_spread_max_over_min_std"] = round(
        float(max(v["std"] for v in num.values()) / min(v["std"] for v in num.values())), 1)

    # --- 5. Categorical cardinality & rare levels ---------------------------
    cat = {}
    for c in CATEGORICAL_COLS:
        s = df[c]
        vc_c = s.value_counts()
        rare = vc_c[vc_c / len(df) < 0.01]
        cat[c] = {
            "n_unique": int(s.nunique()),
            "top_level": str(vc_c.index[0]),
            "top_level_pct": round(100 * float(vc_c.iloc[0] / len(df)), 2),
            "n_levels_under_1pct": int(len(rare)),
            "rows_in_levels_under_1pct": int(rare.sum()),
        }
    audit["categorical"] = cat
    audit["onehot_columns_if_naive"] = int(sum(v["n_unique"] for v in cat.values()))

    # --- 6. Redundancy: education (text) vs education_num (int) -------------
    pair = df.groupby("education", observed=True)["education_num"].nunique()
    audit["education_levels_with_multiple_nums"] = int((pair > 1).sum())
    audit["education_is_perfect_map_of_education_num"] = bool((pair == 1).all())

    # --- 7. Correlation among numerics + relationship to target -------------
    y = (df[TARGET] == ">50K").astype(int)
    corr = df[NUMERIC_COLS].astype(float).corr(method="spearman")
    corr.round(4).to_csv(OUTPUTS / "01_numeric_spearman_corr.csv")
    audit["max_abs_offdiag_numeric_corr"] = round(
        float(corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max()), 4)
    audit["point_biserial_with_target"] = {
        c: round(float(df[c].astype(float).corr(y)), 4) for c in NUMERIC_COLS
    }

    # fnlwgt is a census sampling weight, not a person-level attribute.
    audit["fnlwgt_corr_with_target"] = audit["point_biserial_with_target"]["fnlwgt"]

    OUTPUTS.joinpath("01_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
