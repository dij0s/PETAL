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
    import os
    import io
    import base64
    from pdf2image import convert_from_path
    from pdf2image.exceptions import (
        PDFInfoNotInstalledError,
        PDFPageCountError,
        PDFSyntaxError
    )
    from IPython.display import display
    return base64, convert_from_path, display, io, os


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
def _(convert_from_path):
    def convert_doc_to_images(path):
        yield from convert_from_path(path)
    return (convert_doc_to_images,)


@app.cell
def _(os):
    dataset_directory_path = "./dataset/"
    files = os.listdir(dataset_directory_path)
    return dataset_directory_path, files


@app.cell
def _(convert_doc_to_images, dataset_directory_path, files, os):
    filepath = files[0]
    images = convert_doc_to_images(os.path.join(dataset_directory_path, filepath))
    return (images,)


@app.cell
def _():
    #for index, image in enumerate(images):
    #    if index == 0:
    #        display(image)
    return


@app.cell
def _(images):
    images_list = list(images)
    return (images_list,)


@app.cell
def _(base64, io):
    def get_img_uri(img):
        png_buffer = io.BytesIO()
        img.save(png_buffer, format="PNG")
        png_buffer.seek(0)

        base64_png = base64.b64encode(png_buffer.read()).decode('utf-8')

        data_uri = f"data:image/png;base64,{base64_png}"
        return data_uri
    return (get_img_uri,)


@app.cell
def _(get_img_uri, images, np):
    base64_images = [get_img_uri(img) for img in images]
    np.savez_compressed("./compiled_files.npz", base64_images=base64_images)
    return


@app.cell
def _(AutoProcessor, Qwen2_5_VLForConditionalGeneration):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    return model, processor


@app.cell
def _(model, process_vision_info, processor):
    def analyze_image(data_uri):
        messages = [
            {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": data_uri,
                },
                {"type": "text", "text": "Describe this image."},
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
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text
    return (analyze_image,)


@app.cell
def _(get_img_uri, images_list):
    img = images_list[0]
    data_uri = get_img_uri(img)
    return data_uri, img


@app.cell
def _(display, img):
    display(img)
    return


@app.cell
def _(analyze_image, data_uri):
    res = analyze_image(data_uri)
    return (res,)


@app.cell
def _(res):
    res
    return


if __name__ == "__main__":
    app.run()
