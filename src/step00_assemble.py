"""Step 0 - Assemble the raw UCI files into a single typed CSV.

The two UCI files are not interchangeable:
  * adult.data  : 32,561 rows, no header, labels as '<=50K' / '>50K'
  * adult.test  : 16,282 lines, a junk first line, labels with a trailing '.'
Loading them naively produces four label classes instead of two - a silent bug
that would quietly destroy every downstream metric. We fix it explicitly here.
"""
import json
import pandas as pd
from config import COLUMNS, DATA_RAW, DATA_INTERIM, OUTPUTS

def load_split(filename: str, skiprows: int) -> pd.DataFrame:
    df = pd.read_csv(
        DATA_RAW / filename,
        header=None,
        names=COLUMNS,
        skiprows=skiprows,
        skipinitialspace=True,   # UCI writes ', ' after every comma
        na_values=[],            # we handle '?' ourselves, deliberately
        keep_default_na=False,
    )
    return df

def main() -> None:
    train_raw = load_split("adult.data", skiprows=0)
    test_raw = load_split("adult.test", skiprows=1)  # line 1 is '|1x3 Cross validator'

    report = {
        "adult_data_rows": int(len(train_raw)),
        "adult_test_rows": int(len(test_raw)),
        "labels_before_fix": sorted(
            set(train_raw["income"].unique()) | set(test_raw["income"].unique())
        ),
    }

    # The trailing '.' only exists in adult.test.
    for df in (train_raw, test_raw):
        df["income"] = df["income"].str.strip().str.rstrip(".")

    combined = pd.concat([train_raw, test_raw], ignore_index=True)
    report["labels_after_fix"] = sorted(combined["income"].unique())
    report["combined_rows"] = int(len(combined))
    report["combined_cols"] = int(combined.shape[1])

    # Strip stray whitespace from every text column once, up front.
    obj_cols = combined.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        combined[c] = combined[c].str.strip()

    combined.to_csv(DATA_INTERIM / "adult_raw_combined.csv", index=False)
    (OUTPUTS / "00_assemble.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nWrote {DATA_INTERIM / 'adult_raw_combined.csv'}")

if __name__ == "__main__":
    main()
