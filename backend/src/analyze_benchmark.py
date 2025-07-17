import json
from functools import reduce
import polars as pl

file_template = lambda index: f"benchmarking/sion_benchmark_0{index}.json"
# file_template = lambda index: f"benchmarking/small_benchmark_dp_0{index}.json"
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

df: pl.DataFrame = reduce(
    lambda res, fp: pl.concat([res, pl.DataFrame(read_f(fp))], how="vertical"),
    map(file_template, range(10)),
    pl.DataFrame()
)
if df.count().mean_horizontal().item() != 90.0:
    raise ValueError("Expected 90 benchmarked prompts")

geval_per_sample = (df
    .with_row_index()
    .with_columns(
        (pl.col("index") + 1).mod(10).alias("prompt_index"),
        (pl.col("index").floordiv(9) + 1).alias("benchmark_index"),
        pl.mean_horizontal(*df.columns).alias("geval"),
    )
)

geval_per_prompt = (geval_per_sample
    .group_by("prompt_index")
    .agg(
        pl.col("geval").mean().alias("mean_geval"),
        pl.col("geval").std().alias("std_geval")
    )
)
print(f"This is the geval per prompt: {geval_per_prompt.sort(pl.col('prompt_index'))}")

geval_per_benchmark = (geval_per_sample
    .group_by("benchmark_index")
    .agg(
        pl.col("geval").mean().alias("mean_geval"),
        pl.col("geval").std().alias("std_geval")
    )
)
print(f"This is the geval per benchmark: {geval_per_benchmark.sort(pl.col('benchmark_index'))}")

criteria_stats = (geval_per_sample
    .select(["data_interpretation", "guideline_application", "municipal_relevance", "source_citations"])
    .describe()
)
print(f"This are the statistics per criteria: {criteria_stats}")

# expert scoring
scores = [3, 2, 3, 2, 4, 3, 4, 2, 2]
rescaled_scores = [
    (score - 1) / 4
    for score in scores
]
print(f"These are the expert scores: {rescaled_scores}, mean: {sum(rescaled_scores) / len(rescaled_scores)} and std: {pl.Series(rescaled_scores).std()}")
