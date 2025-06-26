import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The following notebook focuses on parsing and interpreting the infered energy-related datapoints from a typical citizen file as is available in all municipalities.""")
    return


@app.cell
def _():
    import marimo as mo
    import json
    from functools import reduce
    return json, mo, reduce


@app.cell
def _(json):
    with open("./citizen_data_energy_datapoints.json", "r") as f:
        data = json.load(f)
    return (data,)


@app.cell
def _(data):
    list(data.values())[:5]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Parse LLM code-block output into proper Python dictionaries.""")
    return


@app.cell
def _(data, json):
    json_objects = [
        json.loads(d.strip().removeprefix("```json").removesuffix("```<|im_end|>").strip())
        for d in data.values()
    ]
    return (json_objects,)


@app.cell
def _(json_objects):
    json_objects
    return


@app.cell
def _(json_objects, reduce):
    accumulated = reduce(
        lambda res, d: {
            key: value if value is not None and res.get(key) is None else res.get(key)
            for key, value in d.items()
        },
        json_objects,
        {
            "parcel_number": None,
            "sre": None,
            "consumption_heating": None,
            "source_heating": None,
            "power_pv": None
        }
    )
    return (accumulated,)


@app.cell
def _(accumulated):
    accumulated
    return


if __name__ == "__main__":
    app.run()
