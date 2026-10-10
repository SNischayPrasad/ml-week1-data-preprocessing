"""Step 4 - The preprocessing pipeline.

Every transformer here LEARNS something from the data (a median, a mean, a
standard deviation, a category vocabulary). All of it is fitted on the training
split only and then applied unchanged to the test split. That is the whole
point: the test set must stay a stand-in for data the model has never met.

Design decisions, each defended by a measurement from steps 1-3:

  1. Missing categoricals -> their own level "Missing", not the mode.
     Evidence: rows with a missing value earn >50K 13.2% of the time vs 24.8%
     for complete rows. The missingness is informative, so mode-imputation
     would erase a real signal and inject a false one.

  2. Rare categories -> folded into "infrequent" at min_frequency=1%.
     Evidence: native_country has 42 levels, 39 of which cover <1% of rows.
     Holand-Netherlands has exactly ONE row in the whole dataset.

  3. Unknown categories at predict time -> absorbed, never crash.
     Evidence: across 200 random splits, 22.5% produced a test category the
     training split had never seen.

  4. capital_gain / capital_loss -> NO log1p. This reverses our first draft.
     The transform looked obviously correct (skew 7.6 and 4.6; 91.7% and 95.3%
     exact zeros), so we shipped it - then measured it in step 6 and found it
     costs AUC on every model that can notice: -0.00241 for logistic regression,
     -0.00794 for KNN, and exactly 0.00000 for gradient boosting (a monotone
     transform cannot change a tree's splits). Interpretation: the discriminative
     power of capital_gain lives in the size of large gains, and log1p compresses
     precisely that range, while has_capital_gain already carries the zero vs
     non-zero split that the log was really being credited for. Kept as a
     switch, not deleted, so the claim stays falsifiable.

  5. All numerics -> StandardScaler.
     Evidence: raw standard deviations span a 1,479x range. On AUC this is
     nearly free for logistic regression (-0.00016) and exactly free for
     gradient boosting, but it is worth +0.03987 AUC for KNN, and it cuts the
     logistic-regression fit from 17.5 s to 1.5 s by making lbfgs converge.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import DATA_PROCESSED, OUTPUTS, TARGET

SKEWED_NUMERIC = ["capital_gain", "capital_loss"]
PLAIN_NUMERIC = ["age", "education_num", "hours_per_week"]
BINARY_FLAGS = ["capital_gain_topcoded", "has_capital_gain", "has_capital_loss"]
CATEGORICAL = ["workclass", "marital_status", "occupation", "relationship",
               "race", "sex", "native_country"]

MIN_FREQUENCY = 0.01
# Set True to restore the log transform; step 6 measures what that costs.
USE_LOG1P = False


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted preprocessor. Fitting happens in main(), on train only."""

    # Heavy right skew + a zero spike. The textbook move is log1p; we measured
    # it instead of assuming it, and it lost. See decision 4 in the docstring.
    skewed_steps = [("impute", SimpleImputer(strategy="median"))]
    if USE_LOG1P:
        skewed_steps.append(
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one",
                                          validate=True)))
    skewed_steps.append(("scale", StandardScaler()))
    skewed = Pipeline(skewed_steps)

    plain = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical = Pipeline([
        # fill_value makes missingness an explicit, learnable level.
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",  # unseen level -> infrequent bucket
            min_frequency=MIN_FREQUENCY,           # fold the long tail
            sparse_output=False,
            dtype=np.float64,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("skewed", skewed, SKEWED_NUMERIC),
            ("plain", plain, PLAIN_NUMERIC),
            ("flags", "passthrough", BINARY_FLAGS),
            ("cat", categorical, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def main() -> None:
    train = pd.read_csv(DATA_PROCESSED / "train_clean.csv")
    test = pd.read_csv(DATA_PROCESSED / "test_clean.csv")

    X_tr, y_tr = train.drop(columns=[TARGET]), train[TARGET]
    X_te, y_te = test.drop(columns=[TARGET]), test[TARGET]

    pre = build_preprocessor()
    pre.fit(X_tr)                      # <- train only. Never X_te, never the union.
    Z_tr = pre.transform(X_tr)
    Z_te = pre.transform(X_te)
    names = list(pre.get_feature_names_out())

    log = {
        "train_rows": int(Z_tr.shape[0]),
        "test_rows": int(Z_te.shape[0]),
        "features_in": int(X_tr.shape[1]),
        "features_out": int(Z_tr.shape[1]),
        "min_frequency": MIN_FREQUENCY,
    }

    # --- What the encoder actually learned ---------------------------------
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_detail = {}
    for col, cats, infreq in zip(CATEGORICAL, ohe.categories_, ohe.infrequent_categories_):
        n_infreq = 0 if infreq is None else len(infreq)
        cat_detail[col] = {
            "levels_seen_in_train": int(len(cats)),
            "levels_folded_as_infrequent": int(n_infreq),
            "output_columns": int(sum(n.startswith(col + "_") for n in names)),
        }
    log["categorical_encoding"] = cat_detail
    log["onehot_cols_naive"] = int(sum(d["levels_seen_in_train"] for d in cat_detail.values()))
    log["onehot_cols_actual"] = int(sum(d["output_columns"] for d in cat_detail.values()))

    # --- Evidence the scaler was fitted on train only ----------------------
    sc = pre.named_transformers_["plain"].named_steps["scale"]
    log["scaler_fitted_means_train"] = {
        c: round(float(m), 6) for c, m in zip(PLAIN_NUMERIC, sc.mean_)
    }
    log["scaler_means_if_fitted_on_all_data"] = {
        c: round(float(pd.concat([X_tr[c], X_te[c]]).mean()), 6) for c in PLAIN_NUMERIC
    }
    idx = {n: i for i, n in enumerate(names)}
    log["train_mean_after_scaling"] = {
        c: round(float(Z_tr[:, idx[c]].mean()), 9) for c in PLAIN_NUMERIC
    }
    # Test means are NOT 0 - and that is correct. A 0 here would mean leakage.
    log["test_mean_after_scaling"] = {
        c: round(float(Z_te[:, idx[c]].mean()), 6) for c in PLAIN_NUMERIC
    }

    # --- Skew before vs after log1p ----------------------------------------
    log["skew_before_log1p"] = {c: round(float(X_tr[c].skew()), 3) for c in SKEWED_NUMERIC}
    log["skew_after_log1p"] = {
        c: round(float(pd.Series(np.log1p(X_tr[c])).skew()), 3) for c in SKEWED_NUMERIC
    }

    log["nan_remaining_train"] = int(np.isnan(Z_tr).sum())
    log["nan_remaining_test"] = int(np.isnan(Z_te).sum())

    # --- Persist -----------------------------------------------------------
    pd.DataFrame(Z_tr, columns=names).assign(**{TARGET: y_tr.to_numpy()}) \
        .to_csv(DATA_PROCESSED / "train_features.csv", index=False)
    pd.DataFrame(Z_te, columns=names).assign(**{TARGET: y_te.to_numpy()}) \
        .to_csv(DATA_PROCESSED / "test_features.csv", index=False)
    joblib.dump(pre, DATA_PROCESSED / "preprocessor.joblib")
    (OUTPUTS / "04_pipeline.json").write_text(json.dumps(log, indent=2))

    print(json.dumps(log, indent=2))
    print(f"\nFeature names ({len(names)}):")
    print(", ".join(names))


if __name__ == "__main__":
    main()
