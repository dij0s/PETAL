import json
from functools import reduce
import polars as pl

file_template = lambda index: f"benchmarking/sion_benchmark_0{index}.json"
def read_f(filepath) -> list[dict]:
    with open(filepath, "r") as f:
        data = json.load(f)
        return reduce(
            lambda res, x: [
                *res,
                {
                    k: (v - 1) / 4
                    for k, v in x.items()
                    if k != "specific_issues"
                } if "score" not in x else {
                    k: (v - 1) / 4
                    for k, v in x.get("score", {}).items()
                    if k != "specific_issues"
                }
            ],
            data.get("results", []),
            []
        )

df = reduce(
    lambda res, fp: pl.concat([res, pl.DataFrame(read_f(fp))], how="vertical"),
    map(file_template, range(10)),
    pl.DataFrame()
).to_pandas()

prompt_ids = list(range(9)) * 10
df["prompt_id"] = prompt_ids

# Group by prompt (same prompt across runs)
grouped = df.groupby("prompt_id")

# Compute mean and std per prompt
means = grouped.mean(numeric_only=True)
stds = grouped.std(numeric_only=True)

# ---------- Pretty Print ----------
print("=== Per Prompt (Sample): Mean ± Std ===")
for i in means.index:
    print(f"Prompt {i:02}")
    for col in means.columns:
        print(f"  {col:20}: {means.loc[i, col]:.4f} ± {stds.loc[i, col]:.4f}")
    print()

# ---------- Per-feature stats ----------
feature_means = df.mean()
feature_stds = df.std()

print("=== Per Feature: Mean ± Std ===")
for col in df.columns:
    mean = feature_means[col]
    std = feature_stds[col]
    print(f"{col:20}: {mean:.4f} ± {std:.4f}")
