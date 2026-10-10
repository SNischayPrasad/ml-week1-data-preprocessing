"""Central configuration: paths, column groups, and constants.

Keeping these in one place means every script agrees on what a column *is*,
which is the difference between a reproducible pipeline and a pile of notebooks.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "reports" / "figures"

for _d in (DATA_INTERIM, DATA_PROCESSED, OUTPUTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Column names come from adult.names; the UCI files ship without a header row.
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

TARGET = "income"

NUMERIC_COLS = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week",
]

CATEGORICAL_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]

# The UCI files encode missing values as the literal string "?" (with a leading
# space before strip()). pandas will NOT treat these as NaN on its own.
SENTINEL_MISSING = {"?"}
