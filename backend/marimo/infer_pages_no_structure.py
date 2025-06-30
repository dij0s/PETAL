import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The following notebook will serve as base for handling unstructured data that is a municipal file regarding its citizens.

    The different documents that compose it are mostly scanned as digitalization is an ongoing process for these entities.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    from pydantic import BaseModel, Field
    from typing import Optional
    from functools import reduce
    return BaseModel, Field, Optional, mo, reduce


@app.cell
def _():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    return (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        process_vision_info,
    )


@app.cell
def _():
    import numpy as np
    import json
    return json, np


@app.cell
def _(np):
    data = np.load("./citizen_data_energy_export.npz", allow_pickle=True)
    images_list = data["base64_images"]
    return (images_list,)


@app.cell
def _():
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    return max_pixels, min_pixels


@app.cell
def _(
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    max_pixels,
    min_pixels,
):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor


@app.cell
def _(BaseModel, Field, Optional):
    class CitizenData(BaseModel):
        parcel_number: Optional[int] = Field(
            default=None,
            description="Unique numerical identifier of the land or property parcel as listed in administrative or cadastral records. Typically corresponds to the official 'parcel number' assigned by local authorities."
        )
        sre: Optional[float] = Field(
            default=None,
            description="Surface de référence énergétique (SRE) in square meters, representing the energy-relevant reference area of the building, used for calculating energy performance indicators."
        )
        consumption_heating: Optional[float] = Field(
            default=None,
            description="Annual energy consumption for heating purposes, typically expressed in kilowatt-hours (kWh) or per annum (kWh/a). Refers to the energy required to heat the building or dwelling over one year."
        )
        source_heating: Optional[str] = Field(
            default=None,
            description="Primary energy source used for heating. Common values include 'heat pump (PAC)', 'electricity', 'natural gas', 'fuel oil', 'wood', etc. Used to assess environmental and economic impact of heating."
        )
        power_pv: Optional[float] = Field(
            default=None,
            description="Installed peak power of photovoltaic (solar panel) system in kilowatts-peak (kWp) or kilowatt-crête (kWc), indicating the maximum electrical output under standard test conditions."
        )
    return (CitizenData,)


@app.cell
def _(CitizenData, reduce):
    json_schema = reduce(
        lambda res, e: res + f'\n"{e[0]}": {e[1]},',
        CitizenData().model_dump().items(),
        "{"
    ) + "\n}"
    return (json_schema,)


@app.cell
def _(CitizenData, json_schema):
    system_prompt = f"""
    You are a professional energy data retriever and you will be given documents such as building permits or energy assessment for buildings and houses from which you shall retrieve data which suits the following schema :

    {CitizenData.model_json_schema()}

    You must ONLY return valid JSON with the ALL the keys and fill in the values you found in the documents.
    Here's an example output

    {json_schema}

    **IMPORTANT: If you cannot fill a value, simply set it to "None".**
    """
    return (system_prompt,)


@app.cell
def _(system_prompt):
    system_prompt
    return


@app.cell
def _(model, process_vision_info, processor, system_prompt):
    def analyze_image(data_uri):
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": data_uri,
                    },
                    {"type": "text", "text": "Fill in the values from this image and return the JSON."},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=200)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        return output
    return (analyze_image,)


@app.cell
def _(analyze_image, images_list):
    temp = images_list[-1]
    analyze_image(temp)
    return


@app.cell
def _(analyze_image, images_list, json):
    results = {}

    checkpoint_path = "./citizen_data_energy.json"

    for index, data_uri in enumerate(images_list):
        image_id = f"page_{index:04d}"

        try:
            output = analyze_image(data_uri)
            results[image_id] = output
            print(f"Processed {image_id}")
        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            results[image_id] = f"[ERROR] {str(e)}"

        # checkpoint every 10
        # pages and at the end
        if index % 10 == 0 or index == len(images_list) - 1:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Checkpoint saved at page {index}")
    return


if __name__ == "__main__":
    app.run()
