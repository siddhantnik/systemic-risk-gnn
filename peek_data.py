import pandas as pd
import pathlib

ROOT = pathlib.Path("data/raw/interbank_dataset")

for quarter in ["2016_Q1", "2019_Q3"]:
    q_dir = ROOT / quarter
    print(f"\n{'='*60}")
    print(f"QUARTER: {quarter}")
    print(f"{'='*60}")
    
    for fname in ["nodes.csv", "edges.csv"]:
        fpath = q_dir / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            print(f"\n  {fname}: shape={df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  First 3 rows:")
            print(df.head(3).to_string(max_cols=8))
        else:
            print(f"\n  {fname}: NOT FOUND")