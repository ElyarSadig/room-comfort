import argparse
from pathlib import Path
import pandas as pd
import sys

COLS = ["EID", "AbsT", "RelT", "NID", "Temp", "RelH", "L1", "L2", "Occ", "Act", "Door", "Win"]

NUMERIC_COLS = ["EID", "AbsT", "RelT", "NID", "Temp", "RelH", "L1", "L2", "Occ", "Act", "Door", "Win"]


def read_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLS, low_memory=False)
    df["room"] = path.parent.name
    return df


def prepare_dataframe(df: pd.DataFrame, drop_columns: list) -> pd.DataFrame:
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    drop_list = [c for c in drop_columns if c in df.columns]
    if drop_list:
        df = df.drop(columns=drop_list)
        print(f"Dropped columns: {drop_list}")

    df = df.reset_index(drop=True)
    return df


def find_csv_files(base_dir: Path, pattern: str = "*.csv"):
    for p in base_dir.rglob(pattern):
        yield p


def main(args):
    base = Path(args.data_dir)
    if not base.exists():
        print(f"Error: data directory {base} not found", file=sys.stderr)
        sys.exit(1)

    files = list(find_csv_files(base))
    if not files:
        print(f"No CSV files found under {base}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} CSV files. Sample: {files[:5]}")

    dfs = []
    for f in files:
        try:
            df = read_file(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}", file=sys.stderr)

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"Combined shape before cleaning: {combined.shape}")

    combined = prepare_dataframe(combined, args.drop)

    print(f"Combined shape after cleaning: {combined.shape}")
    print("Column dtypes:")
    print(combined.dtypes)

    print("First 5 rows:")
    print(combined.head())

    out = "./combined.csv"
    combined.to_csv(out, index=False)
    print(f"Saved combined CSV to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine Room-Climate CSVs into one DataFrame")
    parser.add_argument("--data-dir", type=str, default="data", help="base data directory (default: data)")
    parser.add_argument(
        "--drop",
        nargs="*",
        default=["EID", "AbsT", "NID", "RelT"],
        help="columns to drop from the combined DataFrame (default: EID AbsT)",
    )
    args = parser.parse_args()
    main(args)