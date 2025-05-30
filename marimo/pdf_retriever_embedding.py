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
    return (np,)


@app.cell
def _(np):
    data = np.load("./compiled_files.npz", allow_pickle=True)
    images_list = data["base64_images"]
    return


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
    You will be provided with an image of a PDF page. The content is written in **French** and originates from either:

    1. Official Swiss legislation, which follows a strict structure (e.g., numbered titles, subtitles, and articles such as "Art. XX"), or  
    2. Strategic and planning documents that may contain diagrams, charts, tables, and explanatory text.

    Your role is to extract and deliver a **literal, and detailed translation** of the content in **English**. **You MUST translate ALL meaningful content from French to English.** This is required for legal and regulatory compliance.

    **Your output must cover the full content of the page but can be summarized to takeway the important information. Do not emit what is important.**

    ---

    ## Context and Purpose:

    The goal is to extract and preserve all important information from each page. The output will later be used in a **Retrieval-Augmented Generation (RAG)** system. This system will respond to user queries to determine **what must or should be done**, based on the content of Swiss legislation and government-designed energy planning.

    Your output must therefore be exhaustive, clear, and suitable for machine indexing and retrieval.

    ---

    ## If the document is legislative (structured):

    - A single page may contain **multiple articles or sections** — include **all of them** in your output.
    - Preserve the hierarchical structure exactly:
      - Titles and subtitles (e.g., "1", "1.1")
      - Article identifiers (e.g., "Art. 4")
      - Paragraphs, bullet points, and numbered clauses

    - Translate the article title into English.
    - Translate the entire content literally into English, with clear formatting to distinguish between sections and articles.
    - DO NOT rephrase legant content, only condense to takeaway important information.

    ---

    ## If the document is unstructured or contains visual elements:

    - Thoroughly describe any **charts, diagrams, plots, or visuals**:
      - Explain the meaning of each component and the relationships depicted.
      - Translate any text, annotations, or labels into English.

    - For **tables**, explain the data clearly and narratively.
      - Example: “The table shows electricity production in 2022: hydropower accounts for 58%, solar for 12%, and nuclear for 20%.”

    - Translate all written text fully.
    - Define technical terms in simple, accessible English when appropriate.
    - If strategic objectives or forecasts are present, explain their meaning and implications.

    ---

    ## General Guidelines:

    - DO NOT mention page numbers, layout, visual formatting, or document type.
    - DO translate all meaningful French content — **literal and complete translation is mandatory**.
    - DO maintain logical structure and section order.
    - DO explain legal, regulatory, or strategic context when evident.
    - DO ensure that the output represents the **entire content of the page**, even if it includes multiple components.

    ---

    ## Output Format:

    {Translated content of the full page}
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

        generated_ids = model.generate(**inputs, max_new_tokens=500, do_sample=False) # do_sample=False equiv. temperature=0
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text
    return


if __name__ == "__main__":
    app.run()
