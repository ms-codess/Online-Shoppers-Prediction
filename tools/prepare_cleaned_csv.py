import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def coerce_bool_series(s: pd.Series) -> pd.Series:
    # Normalize common truthy/falsey strings to booleans
    truthy = {"true", "1", "yes", "y", "t"}
    falsey = {"false", "0", "no", "n", "f"}
    if s.dtype == object:
        lower = s.astype(str).str.strip().str.lower()
        if set(lower.dropna().unique()).issubset(truthy.union(falsey)):
            return lower.map(lambda v: True if v in truthy else (False if v in falsey else np.nan))
    return s


def impute_column(col: pd.Series) -> pd.Series:
    if col.isnull().sum() == 0:
        return col
    if col.dtype == object:
        return col.fillna(col.mode(dropna=True).iloc[0])
    if str(col.dtype) == "bool":
        # Fill missing booleans with mode or False
        mode_vals = col.mode(dropna=True)
        fill_val = bool(mode_vals.iloc[0]) if not mode_vals.empty else False
        return col.fillna(fill_val)
    # Numeric
    return col.fillna(col.median())


def clean_and_encode(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    info = {}
    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    info["dropped_duplicates"] = before - len(df)

    # Coerce boolean-like columns
    for col in df.columns:
        df[col] = coerce_bool_series(df[col])

    # Impute
    for col in df.columns:
        df[col] = impute_column(df[col])

    # Label-encode object columns (notebook-style)
    le_map = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_map[col] = list(le.classes_)

    # Convert booleans to ints
    for col in df.columns:
        if str(df[col].dtype) == "bool":
            df[col] = df[col].astype(int)

    return df, {"dropped_duplicates": info.get("dropped_duplicates", 0), "label_maps": le_map}


def main():
    ap = argparse.ArgumentParser(description="Prepare cleaned CSV like the notebook steps.")
    ap.add_argument("--input", default="online_shoppers_intention.csv", help="Path to raw CSV")
    ap.add_argument("--output", default="online_shoppers_intention_cleaned.csv", help="Output path for cleaned CSV")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    cleaned, meta = clean_and_encode(df)
    cleaned.to_csv(args.output, index=False)

    print(f"Saved cleaned dataset to: {os.path.abspath(args.output)}")
    print(f"Rows: {len(cleaned)}, Columns: {len(cleaned.columns)}")
    print(f"Dropped duplicates: {meta['dropped_duplicates']}")
    if meta["label_maps"]:
        print("Encoded columns:")
        for c, classes in meta["label_maps"].items():
            print(f" - {c}: {len(classes)} categories")


if __name__ == "__main__":
    main()

