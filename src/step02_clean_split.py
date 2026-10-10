"""Step 2 - Cleaning decisions, then a stratified split.

Ordering is deliberate and is itself a result:
  dedupe -> drop leaky/useless columns -> sentinel to NaN -> SPLIT
Everything that *learns* from data (imputation values, scaler statistics,
encoder vocabularies) happens AFTER the split, in step 4, fit on train only.

Deduplicating BEFORE the split matters: 52 identical rows split at random would
put copies of the same record on both sides of the fence, inflating test scores.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import (DATA_INTERIM, DATA_PROCESSED, OUTPUTS, RANDOM_STATE,
                    SENTINEL_MISSING, TARGET, TEST_SIZE)

DROP_COLS = ["fnlwgt", "education"]
CAPITAL_GAIN_TOPCODE = 99999

def main() -> None:
    df = pd.read_csv(DATA_INTERIM / "adult_raw_combined.csv", skipinitialspace=True)
    log = {"rows_in": int(len(df)), "cols_in": int(df.shape[1])}

    # --- 2.1 Duplicates -----------------------------------------------------
    feat_cols = [c for c in df.columns if c != TARGET]
    contradictory = df.duplicated(subset=feat_cols, keep=False) & ~df.duplicated(keep=False)
    log["contradictory_label_rows"] = int(contradictory.sum())
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log["exact_duplicates_dropped"] = int(before - len(df))
    log["rows_after_dedupe"] = int(len(df))

    # --- 2.2 Drop columns that cannot help ----------------------------------
    # fnlwgt: a census post-stratification weight describing how many people the
    #   row represents - a property of the survey, not the person. |r| = 0.006.
    # education: a perfect 1-to-1 relabelling of education_num (verified in
    #   step 1). Keeping both adds 16 one-hot columns carrying zero new signal
    #   and destroys the natural ordering.
    df = df.drop(columns=DROP_COLS)
    log["columns_dropped"] = DROP_COLS
    log["cols_after_drop"] = int(df.shape[1])

    # --- 2.3 Sentinel '?' -> real NaN ---------------------------------------
    sentinel_total = 0
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            n = int(df[c].isin(SENTINEL_MISSING).sum())
            sentinel_total += n
            df[c] = df[c].replace(list(SENTINEL_MISSING), np.nan)
    log["sentinel_values_converted_to_nan"] = sentinel_total
    log["nan_after_conversion"] = int(df.isna().sum().sum())

    # --- 2.4 Top-coding flag (do NOT delete the rows) -----------------------
    topcoded = (df["capital_gain"] == CAPITAL_GAIN_TOPCODE)
    log["capital_gain_topcoded_rows"] = int(topcoded.sum())
    df["capital_gain_topcoded"] = topcoded.astype(int)
    # Preserve "very large" without letting 99999 dominate a scaler: replace the
    # censoring code with the largest genuinely observed value.
    true_max = int(df.loc[~topcoded, "capital_gain"].max())
    log["capital_gain_true_max_observed"] = true_max
    df.loc[topcoded, "capital_gain"] = true_max

    # --- 2.5 Engineered flags for the zero-inflated capital columns ---------
    df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)
    log["engineered_features"] = ["capital_gain_topcoded", "has_capital_gain", "has_capital_loss"]

    # --- 2.6 Target -> binary int -------------------------------------------
    df[TARGET] = (df[TARGET] == ">50K").astype(int)
    log["positive_rate_pct"] = round(100 * float(df[TARGET].mean()), 3)

    # --- 2.7 Stratified split ----------------------------------------------
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    log["train_rows"] = int(len(X_tr))
    log["test_rows"] = int(len(X_te))
    log["train_positive_rate_pct"] = round(100 * float(y_tr.mean()), 3)
    log["test_positive_rate_pct"] = round(100 * float(y_te.mean()), 3)

    # Unseen-category check: does any test level not occur in train?
    unseen = {}
    for c in X.select_dtypes(include=["object", "string"]).columns:
        miss = set(X_te[c].dropna().unique()) - set(X_tr[c].dropna().unique())
        if miss:
            unseen[c] = sorted(map(str, miss))
    log["categories_in_test_absent_from_train"] = unseen

    X_tr.assign(**{TARGET: y_tr}).to_csv(DATA_PROCESSED / "train_clean.csv", index=False)
    X_te.assign(**{TARGET: y_te}).to_csv(DATA_PROCESSED / "test_clean.csv", index=False)
    OUTPUTS.joinpath("02_clean_split.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))

if __name__ == "__main__":
    main()
