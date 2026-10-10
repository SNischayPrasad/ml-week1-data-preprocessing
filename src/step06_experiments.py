"""Step 6 - Ablations. Each decision from step 4 is re-run with that one knob
changed and everything else held fixed, so the write-up can quote an effect size
instead of an opinion.

Primary metric is ROC-AUC: the classes are imbalanced 3.18:1, so accuracy is
close to meaningless here (predicting "<=50K" for everybody already scores
76.1%). PR-AUC and F1 for the positive class are reported alongside.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import DATA_PROCESSED, OUTPUTS, RANDOM_STATE, TARGET
from step04_pipeline import (BINARY_FLAGS, CATEGORICAL, PLAIN_NUMERIC,
                             SKEWED_NUMERIC)

warnings.filterwarnings("ignore", category=UserWarning)

train = pd.read_csv(DATA_PROCESSED / "train_clean.csv")
test = pd.read_csv(DATA_PROCESSED / "test_clean.csv")
X_tr, y_tr = train.drop(columns=[TARGET]), train[TARGET]
X_te, y_te = test.drop(columns=[TARGET]), test[TARGET]

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}


def make_pre(impute="constant", min_freq=0.01, scale=True, log_skew=False):
    """Preprocessor with one knob per ablation."""
    skewed_steps = [("impute", SimpleImputer(strategy="median"))]
    if log_skew:
        skewed_steps.append(("log1p", FunctionTransformer(np.log1p, validate=True)))
    if scale:
        skewed_steps.append(("scale", StandardScaler()))

    plain_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        plain_steps.append(("scale", StandardScaler()))

    if impute == "constant":
        cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
    else:
        cat_imputer = SimpleImputer(strategy="most_frequent")

    ohe_kw = dict(handle_unknown="infrequent_if_exist", sparse_output=False)
    if min_freq is not None:
        ohe_kw["min_frequency"] = min_freq
    else:
        ohe_kw["handle_unknown"] = "ignore"

    cat_steps = [("impute", cat_imputer), ("onehot", OneHotEncoder(**ohe_kw))]

    return ColumnTransformer([
        ("skewed", Pipeline(skewed_steps), SKEWED_NUMERIC),
        ("plain", Pipeline(plain_steps), PLAIN_NUMERIC),
        ("flags", "passthrough", BINARY_FLAGS),
        ("cat", Pipeline(cat_steps), CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=False)


def model(kind):
    if kind == "logreg":
        return LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=25, n_jobs=-1)
    raise ValueError(kind)


def evaluate(pre, kind="logreg", Xtr=X_tr, ytr=y_tr, Xte=X_te, yte=y_te,
             do_cv=True, label=""):
    t0 = time.time()
    pipe = Pipeline([("pre", pre), ("clf", model(kind))])
    cv_auc = cv_std = None
    if do_cv:
        scores = cross_val_score(pipe, Xtr, ytr, cv=CV, scoring="roc_auc", n_jobs=-1)
        cv_auc, cv_std = float(np.mean(scores)), float(np.std(scores))
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    r = {
        "cv_auc_train": round(cv_auc, 5) if cv_auc is not None else None,
        "cv_auc_std": round(cv_std, 5) if cv_std is not None else None,
        "test_auc": round(float(roc_auc_score(yte, proba)), 5),
        "test_pr_auc": round(float(average_precision_score(yte, proba)), 5),
        "test_f1_positive": round(float(f1_score(yte, pred)), 5),
        "test_accuracy": round(float((pred == yte).mean()), 5),
        "n_features": int(pipe.named_steps["pre"].transform(Xtr.head(50)).shape[1]),
        "seconds": round(time.time() - t0, 1),
    }
    if label:
        print(f"  {label:<46s} AUC={r['test_auc']:.5f}  F1={r['test_f1_positive']:.4f}  "
              f"({r['seconds']}s)")
    return r


def main() -> None:
    run_all()


def run_all() -> None:
    # ===== Baselines ========================================================
    print("\n[0] Baselines")
    maj = {
        "strategy": "always predict <=50K",
        "test_accuracy": round(float((y_te == 0).mean()), 5),
        "test_auc": 0.5,
        "test_f1_positive": 0.0,
    }
    print(f"  majority-class accuracy = {maj['test_accuracy']:.5f}, AUC = 0.5")
    results["baseline_majority"] = maj

    # ===== A. Missing-value strategy ===========================================
    print("\n[A] Missing-value strategy (logreg, everything else fixed)")
    results["A_missing"] = {
        "explicit_Missing_category": evaluate(make_pre(impute="constant"), label="explicit 'Missing' category"),
        "most_frequent_imputation": evaluate(make_pre(impute="most_frequent"), label="most-frequent (mode) imputation"),
    }
    # Dropping the rows entirely - note this also changes the training set size.
    m_tr = X_tr.notna().all(axis=1)
    m_te = X_te.notna().all(axis=1)
    r_drop = evaluate(make_pre(impute="constant"), Xtr=X_tr[m_tr], ytr=y_tr[m_tr],
                      label="drop rows containing any NaN")
    r_drop["train_rows_used"] = int(m_tr.sum())
    r_drop["train_rows_discarded"] = int((~m_tr).sum())
    r_drop["note"] = "evaluated on the FULL test set, so the comparison stays fair"
    results["A_missing"]["drop_incomplete_rows"] = r_drop

    # ===== B. Rare-category folding ============================================
    print("\n[B] Rare-level folding")
    results["B_rare_levels"] = {
        "min_frequency_1pct": evaluate(make_pre(min_freq=0.01), label="min_frequency = 1%"),
        "naive_one_hot": evaluate(make_pre(min_freq=None), label="naive one-hot (every level kept)"),
    }

    # ===== C. Scaling, across model families ===================================
    print("\n[C] Scaling on/off, by model family")
    scaling = {}
    knn_tr = X_tr.sample(n=8000, random_state=RANDOM_STATE)
    knn_y = y_tr.loc[knn_tr.index]
    for kind in ["logreg", "hgb", "knn"]:
        sub = {}
        for flag, name in [(True, "scaled"), (False, "unscaled")]:
            if kind == "knn":
                sub[name] = evaluate(make_pre(scale=flag), kind, Xtr=knn_tr, ytr=knn_y,
                                     do_cv=False, label=f"{kind:<7s} {name} (8k subsample)")
            else:
                sub[name] = evaluate(make_pre(scale=flag), kind, do_cv=False,
                                     label=f"{kind:<7s} {name}")
        sub["auc_gain_from_scaling"] = round(sub["scaled"]["test_auc"] - sub["unscaled"]["test_auc"], 5)
        scaling[kind] = sub
    results["C_scaling"] = scaling
    results["C_scaling"]["knn_note"] = "KNN uses an 8,000-row training subsample for runtime; both arms use the identical subsample."

    # ===== D. log1p on the skewed capital columns ==============================
    print("\n[D] log1p on capital_gain / capital_loss")
    results["D_log1p"] = {
        "with_log1p": evaluate(make_pre(log_skew=True), label="with log1p"),
        "without_log1p": evaluate(make_pre(log_skew=False), label="without log1p"),
    }

    # ===== D2. Is the log1p penalty model-specific? ==========================
    # A monotone transform cannot change a tree's splits, so gradient boosting
    # should show EXACTLY zero. That doubles as a check that this harness works.
    print("\n[D2] log1p effect by model family")
    by_model = {}
    for kind in ["logreg", "knn", "hgb"]:
        Xa, ya = (knn_tr, knn_y) if kind == "knn" else (X_tr, y_tr)
        for flag in (True, False):
            r = evaluate(make_pre(log_skew=flag), kind, Xtr=Xa, ytr=ya, do_cv=False)
            by_model[f"{kind}_{'log1p' if flag else 'raw'}"] = r["test_auc"]
        delta = by_model[f"{kind}_log1p"] - by_model[f"{kind}_raw"]
        by_model[f"{kind}_delta"] = round(delta, 5)
        print(f"  {kind:<7s} log1p={by_model[f'{kind}_log1p']:.5f}  "
              f"raw={by_model[f'{kind}_raw']:.5f}  delta={delta:+.5f}")
    (OUTPUTS / "06b_log1p_by_model.json").write_text(json.dumps(by_model, indent=2))
    results["D2_log1p_by_model"] = by_model

    # ===== E. Leakage =========================================================
    print("\n[E] Data leakage, three magnitudes")
    leak = {}

    # E1 - scaler/encoder statistics fitted on train+test instead of train only.
    pre_leaky = make_pre()
    pre_leaky.fit(pd.concat([X_tr, X_te], ignore_index=True))   # the mistake
    clf = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    clf.fit(pre_leaky.transform(X_tr), y_tr)
    auc_leaky = roc_auc_score(y_te, clf.predict_proba(pre_leaky.transform(X_te))[:, 1])
    auc_clean = results["A_missing"]["explicit_Missing_category"]["test_auc"]
    leak["E1_scaler_fitted_on_train_plus_test"] = {
        "auc_leaky": round(float(auc_leaky), 5),
        "auc_correct": auc_clean,
        "optimism": round(float(auc_leaky) - auc_clean, 5),
        "note": "Small, because StandardScaler statistics are stable at n=48,790. "
                "The discipline still matters - the same mistake is large below.",
    }
    print(f"  E1 scaler fit on train+test        optimism = {leak['E1_scaler_fitted_on_train_plus_test']['optimism']:+.5f} AUC")

    # E2 - target encoding a high-cardinality key, fitted before vs after the split.
    def target_encode(key_tr, y_fit_src_key, y_fit_src_y, key_apply, prior, k=20):
        stats = pd.DataFrame({"k": y_fit_src_key, "y": y_fit_src_y}).groupby("k")["y"].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + prior * k) / (stats["count"] + k)
        return key_apply.map(smooth).fillna(prior).to_numpy()

    key_cols = ["occupation", "native_country", "workclass", "education_num", "race"]
    key_tr = X_tr[key_cols].fillna("NA").astype(str).agg("|".join, axis=1)
    key_te = X_te[key_cols].fillna("NA").astype(str).agg("|".join, axis=1)
    all_key = pd.concat([key_tr, key_te], ignore_index=True)
    all_y = pd.concat([y_tr, y_te], ignore_index=True)
    prior = float(y_tr.mean())

    # leaky: test labels contribute to the encoding applied to test rows
    te_tr_leak = target_encode(None, all_key, all_y, key_tr, prior)
    te_te_leak = target_encode(None, all_key, all_y, key_te, prior)
    # clean: encoding learned on train labels only
    te_tr_ok = target_encode(None, key_tr, y_tr, key_tr, prior)
    te_te_ok = target_encode(None, key_tr, y_tr, key_te, prior)

    def auc_single(f_tr, f_te):
        m = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        m.fit(f_tr.reshape(-1, 1), y_tr)
        return float(roc_auc_score(y_te, m.predict_proba(f_te.reshape(-1, 1))[:, 1]))

    leak["E2_target_encoding_before_split"] = {
        "key_cardinality": int(all_key.nunique()),
        "auc_leaky": round(auc_single(te_tr_leak, te_te_leak), 5),
        "auc_correct": round(auc_single(te_tr_ok, te_te_ok), 5),
        "optimism": round(auc_single(te_tr_leak, te_te_leak) - auc_single(te_tr_ok, te_te_ok), 5),
        "note": "Single-feature model, so the leak is visible in isolation.",
    }
    print(f"  E2 target encoding before split    optimism = {leak['E2_target_encoding_before_split']['optimism']:+.5f} AUC "
          f"(key cardinality {leak['E2_target_encoding_before_split']['key_cardinality']:,})")

    # E3 - duplicates left in place, then split at random.
    raw_comb = pd.read_csv(DATA_PROCESSED.parent / "interim" / "adult_raw_combined.csv",
                           skipinitialspace=True)
    dups = int(raw_comb.duplicated().sum())
    leak["E3_duplicates_across_split"] = {
        "exact_duplicate_rows": dups,
        "share_of_dataset_pct": round(100 * dups / len(raw_comb), 4),
        "note": "Deduplicated BEFORE splitting, so no record can appear on both sides. "
                "At 0.11% of rows the metric impact is negligible here, but the ordering "
                "is what generalises to datasets where duplication is heavy.",
    }
    results["E_leakage"] = leak

    # ===== F. Final model on the chosen pipeline ===============================
    print("\n[F] Final pipeline")
    results["F_final"] = {
        "logreg": evaluate(make_pre(), "logreg", label="logistic regression"),
        "hist_gradient_boosting": evaluate(make_pre(), "hgb", label="hist gradient boosting"),
    }

    (OUTPUTS / "06_experiments.json").write_text(json.dumps(results, indent=2))
    print("\nWrote outputs/06_experiments.json")


if __name__ == "__main__":
    main()
