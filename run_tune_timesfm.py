#!/usr/bin/env python3
"""Quick standalone script to tune TimesFM hyperparameters only."""
import pandas as pd
from tune_hyperparams import tune_timesfm

df = pd.read_csv("data/601933_10yr.csv")
print(f"Loaded {len(df)} rows")

results = tune_timesfm(df, train_ratio=0.6, top_k=5)

print("\nTop 5 TimesFM configs (by val Sharpe):")
for i, r in enumerate(results, 1):
    p = r["params"]
    print(f"  {i}. ctx={p['context_len']}, buy_pct={p['buy_pct']}, "
          f"spread={p['strong_spread']}  "
          f"sharpe={r['sharpe_ratio']:.3f}  ret={r['total_return']:+.2f}%")
