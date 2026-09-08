"""Shared cleaning and feature-encoding code.

Used by notebooks/01_cleaning.ipynb, notebooks/02_modelling.ipynb,
src/train.py and app/app.py so that all of them agree on the pipeline.

Run ``python src/features.py`` to rebuild data/cleaned_item_details.csv
from data/raw/item_details.csv.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "item_details.csv"
CLEAN_PATH = REPO_ROOT / "data" / "cleaned_item_details.csv"

# Column names as scraped from divar.ir (Persian) -> names used in this repo.
COLUMN_MAP = {
    "برند و مدل": "Brand and Model",
    "وضعیت": "Status",
    "تعداد سیم‌کارت": "SIM Count",
    "اصالت برند": "Brand Origin",
    "حافظهٔ داخلی": "Internal Storage(GB)",
    "مقدار رم": "RAM(GB)",
    "قیمت": "Price(Toman)",
    "رنگ": "Color",
}

TARGET = "Price(Toman)"
CATEGORICAL = ["Brand", "Status", "Brand Origin", "Color"]
NUMERIC = ["SIM Count", "Internal Storage(GB)", "RAM(GB)"]
FEATURES = CATEGORICAL + NUMERIC

# Listings priced outside this range are placeholders or typos, not prices.
# On Divar a price of 1,000 Toman means "contact me"; the top end removes a
# handful of entries such as an iPhone 4 listed at 48 billion Toman.
PRICE_MIN = 500_000
PRICE_MAX = 300_000_000

NOT_SPECIFIED = "مطرح نیست"

_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٫٬", "0123456789..")
_UNIT_TO_GB = {"ترابایت": 1000.0, "گیگابایت": 1.0, "مگابایت": 1 / 1000}


def to_ascii_digits(text: str) -> str:
    """Translate Persian digits and separators to ASCII."""
    return str(text).translate(_PERSIAN_DIGITS)


def parse_capacity_gb(text: str) -> float:
    """'۲۵۶ گیگابایت' -> 256.0, '۱ ترابایت' -> 1000.0, '۵۱۲ مگابایت' -> 0.512."""
    ascii_text = to_ascii_digits(text)
    number = float(re.search(r"[\d.]+", ascii_text).group())
    for unit, factor in _UNIT_TO_GB.items():
        if unit in text:
            return number * factor
    raise ValueError(f"Unknown capacity unit in {text!r}")


def parse_sim_count(text: str) -> int:
    """'۲ عدد' -> 2, '۳ و بیشتر' (3 or more) -> 3."""
    return int(re.search(r"\d+", to_ascii_digits(text)).group())


def parse_price_toman(text: str) -> int:
    """'۳۹٬۰۰۰٬۰۰۰ تومان' -> 39000000."""
    digits = re.sub(r"\D", "", to_ascii_digits(text))
    return int(digits)


def normalise_color(text: str) -> str:
    """Drop Latin words, collapse whitespace variants ('آبی ' and 'آبی' were
    separate classes in the original notebook)."""
    text = re.sub(r"[A-Za-z]+", "", str(text))
    return " ".join(text.split())


def extract_brand(brand_and_model: pd.Series) -> pd.Series:
    """First token of 'Brand and Model', e.g. 'سامسونگ Galaxy A7' -> 'سامسونگ'."""
    return brand_and_model.astype(str).str.split().str[0]


def clean(raw: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Turn the scraped CSV into the modelling table.

    Steps (row counts for the shipped dataset in brackets):
      1. drop rows with any missing field                      (3221 -> 2165)
      2. drop rows where a field is "not specified"            (-> 2150)
      3. parse Persian numbers and units into numeric columns
      4. normalise colour text and drop rows with no colour left
      5. drop exact duplicate rows (the scraper revisited listings)
      6. keep prices in [PRICE_MIN, PRICE_MAX]
      7. add a Brand column
    """
    counts = {"scraped": len(raw)}
    df = raw.rename(columns=COLUMN_MAP).dropna()
    counts["after dropna"] = len(df)

    df = df[~df.astype(str).apply(lambda col: col.str.contains(NOT_SPECIFIED)).any(axis=1)]
    counts["after 'not specified' filter"] = len(df)

    df = df.assign(
        **{
            "SIM Count": df["SIM Count"].map(parse_sim_count),
            "Internal Storage(GB)": df["Internal Storage(GB)"].map(parse_capacity_gb),
            "RAM(GB)": df["RAM(GB)"].map(parse_capacity_gb),
            TARGET: df[TARGET].map(parse_price_toman),
            "Color": df["Color"].map(normalise_color),
        }
    )
    df = df[df["Color"] != ""]
    counts["after colour filter"] = len(df)

    df = df.drop_duplicates()
    counts["after dropping duplicates"] = len(df)

    df = df[df[TARGET].between(PRICE_MIN, PRICE_MAX)]
    counts["after price range filter"] = len(df)

    df = df.assign(Brand=extract_brand(df["Brand and Model"]))
    df = df[["Brand and Model", "Brand", "Status", "Brand Origin", "Color", *NUMERIC, TARGET]]

    if verbose:
        for step, n in counts.items():
            print(f"{step:32s} {n:5d}")
    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode the categorical columns, standardise the numeric ones.

    Fit this on the training split only; unseen categories at prediction
    time become all-zero rows instead of raising.
    """
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    df = clean(raw, verbose=True)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False, lineterminator="\n")
    print(f"wrote {len(df)} rows to {CLEAN_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
