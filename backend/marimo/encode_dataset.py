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
    from pdfminer.high_level import extract_text
    from IPython.display import display
    return base64, convert_from_path, extract_text, io, os


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
def _(extract_text):
    def extract_text_from_doc(path):
        text = extract_text(path)
        return text
    return


@app.cell
def _(os):
    dataset_directory_path = "./dataset/"
    files = os.listdir(dataset_directory_path)
    return dataset_directory_path, files


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
def _(
    convert_doc_to_images,
    dataset_directory_path,
    files,
    get_img_uri,
    np,
    os,
):
    base64_images = [get_img_uri(img) for filename in files for img in convert_doc_to_images(os.path.join(dataset_directory_path, filename))]
    np.savez_compressed("./compiled_files.npz", base64_images=base64_images)
    return


if __name__ == "__main__":
    app.run()
