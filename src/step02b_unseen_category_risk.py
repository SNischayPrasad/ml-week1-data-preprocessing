"""Step 2b - How often does a random split hand the encoder an unseen category?

`native_country` has 42 levels and 'Holand-Netherlands' occurs exactly ONCE in
all 48,842 rows. With random_state=42 it happens to land in the training split,
so the problem is invisible - which is precisely why it is worth measuring
instead of trusting. We repeat the split across many seeds and count.

The answer sets the encoder's handle_unknown policy in step 4.
"""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_INTERIM, OUTPUTS, RANDOM_STATE, TARGET, TEST_SIZE

N_SEEDS = 200


def main() -> None:
    df = pd.read_csv(DATA_INTERIM / "adult_raw_combined.csv", skipinitialspace=True)
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = (df[TARGET] == ">50K").astype(int)

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    cat_cols = X.select_dtypes(include=["object", "string"]).columns

    hits, detail = 0, {}
    for seed in range(N_SEEDS):
        X_tr, X_te, _, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
        for c in cat_cols:
            unseen = set(X_te[c].dropna().unique()) - set(X_tr[c].dropna().unique())
            if unseen:
                hits += 1
                for u in unseen:
                    detail[str(u)] = detail.get(str(u), 0) + 1
                break

    # Was the seed we actually ship one of the unlucky ones?
    X_tr, X_te, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    shipped_unseen = any(
        set(X_te[c].dropna().unique()) - set(X_tr[c].dropna().unique()) for c in cat_cols)

    rarest = df["native_country"].value_counts().tail(1)
    report = {
        "seeds_tested": N_SEEDS,
        "seeds_with_unseen_test_category": hits,
        "pct": round(100 * hits / N_SEEDS, 1),
        "levels_involved": detail,
        "rarest_level": str(rarest.index[0]),
        "rarest_level_row_count": int(rarest.iloc[0]),
        "seed_42_had_unseen": bool(shipped_unseen),
    }
    (OUTPUTS / "02b_unseen_category_risk.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
