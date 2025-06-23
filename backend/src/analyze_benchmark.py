import json

with open("benchmarking/sion_benchmarking_deepseek.json", "r") as f:
    data = json.load(f)

from functools import reduce

normalized_scores = [
    (
        scores.get("data_interpretation", 0)
        + scores.get("guideline_application", 0)
        + scores.get("municipal_relevance", 0)
        + scores.get("source_citations", 0)
    ) / 4 / 5
    for scores in map(lambda d: d.get("score", {}), data)
]
mean_normalized_score = sum(normalized_scores) / len(normalized_scores)
std_normalized_score = (
    sum(
        (x - mean_normalized_score)**2
        for x in normalized_scores
    ) / (len(normalized_scores) - 1)
) ** 0.5
print(f"Normalized scores: {normalized_scores}, average score: {mean_normalized_score:.2f} ± {std_normalized_score:.2f}")

times = [
    d.get("time", 0)
    for d in data
]
mean_time = sum(times) / len(times)
std_time = (
    sum(
        (x - mean_time)**2
        for x in times
    ) / (len(times) - 1)
) ** 0.5
print(f"Times: {times}, average time: {mean_time:.2f} ± {std_time:.2f}")
