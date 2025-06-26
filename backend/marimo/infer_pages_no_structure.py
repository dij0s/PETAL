import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The following notebook will serve as base for handling and embedding PDF files and their multimodal content (text and images) for further use in RAG-application context.

    The documents are policies, prescriptions and "design" documents related to the energy planning and transition, in Switzerland.

    Sources:
    https://cookbook.openai.com/examples/parse_pdf_docs_for_rag

    Le traitement de documents PDF nécessite l'installation du programme ```poppler``` sur l'hôte (https://pypi.org/project/pdf2image/).
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


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
    data = np.load("./citizen_data_export.npz", allow_pickle=True)
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
def _():
    system_prompt = """
    You will receive an image of a document page in French, originating from municipal citizen data (scans or digitalized documents).
    Your task is to extract and deliver a **literal, complete, and detailed English translation** of all meaningful content on the page.

    **CRITICAL ANONYMIZATION REQUIREMENTS - ABSOLUTE PRIORITY:**
    - **MANDATORY: Replace ALL personal identifiers without exception** - names (first, last, maiden, nicknames), addresses (street, city, postal codes), phone numbers, email addresses, social security numbers, ID numbers, passport numbers, driver's license numbers, tax identification numbers, bank account details, medical record numbers, employee IDs, student IDs, membership numbers, license plates, and ANY other identifying information.
    - **MANDATORY: Anonymize ALL dates** - birth dates, appointment dates, registration dates, expiration dates, etc. Replace with [DATE_REDACTED] or generic placeholders like [BIRTH_DATE], [REGISTRATION_DATE].
    - **MANDATORY: Replace ALL signatures, initials, or handwritten personal marks** with [SIGNATURE_REDACTED].
    - **MANDATORY: Replace company names, organization names, and institutional affiliations** with generic placeholders like [COMPANY_NAME], [ORGANIZATION], [INSTITUTION].
    - **MANDATORY: Anonymize financial information** - account numbers, transaction amounts, salaries, debts - replace with [FINANCIAL_DATA_REDACTED].
    - **MANDATORY: Redact ANY reference numbers, case numbers, file numbers, or tracking codes** that could be used to trace back to individuals.
    - **MANDATORY: Replace geographical specifics** beyond general region/city type with [LOCATION_REDACTED].
    - **MANDATORY: Use consistent placeholder formatting** - always use square brackets with descriptive labels like [PERSONAL_NAME], [HOME_ADDRESS], [PHONE_NUMBER], [EMAIL_ADDRESS], [ID_NUMBER].

    **SECONDARY REQUIREMENTS:**
    - Translate **all written text fully and literally** without omitting important details.
    - Describe any charts, tables, or diagrams clearly, translating all labels and data into English while maintaining anonymization.
    - Maintain logical structure and section order.
    - Do NOT mention page numbers or formatting details.
    - Ensure the output is exhaustive and suitable for machine indexing in a retrieval system.

    **VERIFICATION CHECKLIST - Before finalizing output, confirm:**
    ✓ No real names remain visible
    ✓ No actual addresses are present
    ✓ No genuine dates are displayed
    ✓ No contact information is exposed
    ✓ No identification numbers are shown
    ✓ All personal data uses consistent [PLACEHOLDER] format

    **REMEMBER: Privacy protection is the absolute top priority. When in doubt, REDACT.**

    Output format:  
    {Full anonymized, translated content of the page in English}
    """
    return (system_prompt,)


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
                    {"type": "text", "text": "Describe, in detail, this image."},
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

        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False) # do_sample=False equiv. temperature=0
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text
    return (analyze_image,)


@app.cell
def _(analyze_image, images_list, json):
    results = {}

    checkpoint_path = "./citizen_data_anonimised.json"

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
