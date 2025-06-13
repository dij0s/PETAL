import json
import re
from functools import reduce

with open("./marimo/documents_with_visual_analysis.json", "r") as file:
    data = json.load(file)

# extract all key figures
valid_units = [
    'GWh', 'GWhth', 'GWhel', 'GWh/a', 'GWh/month', 'GWh/year', 'GWh/hivernale'
]

unit_pattern = '|'.join(re.escape(unit) for unit in valid_units)
regex = rf'\b(\d+(?:[\.,]\d+)?)\s+(?:{unit_pattern})\b(?!\s+per\b)'
matches = reduce(
    lambda res, d: [
        *res,
        *re.findall(regex, d["visual_analysis"]["analysis"])
    ],
    data.values(),
    []
)

def clean_numerical_matches(match: str) -> float:
    return float(match.replace(',', '').lstrip('0') or '0')

SCALE_FACTOR = 0.05

print([clean_numerical_matches(match) * SCALE_FACTOR  for match in matches])
