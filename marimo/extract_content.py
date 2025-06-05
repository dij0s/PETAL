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
    import json
    import base64
    import pymupdf
    import numpy as np
    from PIL import Image
    from functools import reduce
    return base64, os, pymupdf, reduce


@app.cell
def _():
    from layoutparser.models import Detectron2LayoutModel
    return (Detectron2LayoutModel,)


@app.cell
def _():
    DATASET_PATH = "./dataset"
    OUTPUT_JSON_PATH = "./extracted_pages.json"
    return (DATASET_PATH,)


@app.cell
def _(base64):
    def pixmap_to_base64(pixmap, image_format="PNG"):
        image_bytes = pixmap.tobytes(output=image_format)
        base64_bytes = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/{image_format.lower()};base64,{base64_bytes}"
    return (pixmap_to_base64,)


@app.cell
def _(pixmap_to_base64, pymupdf, reduce):
    def extract_content(pdf_path):
        doc = pymupdf.open(pdf_path)
        page_content = []

        for index, page in enumerate(doc):
            page_data = {
                "page_number": index + 1,
                "pixmap_render": pixmap_to_base64(page.get_pixmap(dpi=300)),
                "content": reduce(
                    lambda res, t: f"{res}\n{t.strip()}" if t.strip() else res,
                    map(lambda block: block[4], page.get_text("blocks")),
                    ""
                ),
            }
            page_content.append(page_data)

        return page_content
    return (extract_content,)


@app.cell
def _(Detectron2LayoutModel):
    model = Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')
    def segment_with_layoutparser(page_img_pil):
        layout = model.detect(page_img_pil)
        segments = []
        for block in layout:
            if block.type in ['Figure', 'Table']:  # or whatever you want
                x1, y1, x2, y2 = map(int, block.coordinates)
                crop = page_img_pil.crop((x1, y1, x2, y2))
                segments.append(crop)
        return segments
    return


@app.cell
def _(DATASET_PATH, extract_content, os, reduce):
    documents = reduce(
        lambda res, filename: [*res, {
            "filename": filename,
            "pages": extract_content(os.path.join(DATASET_PATH, filename))
        }],
        filter(lambda filename: filename.lower().endswith(".pdf"), os.listdir(DATASET_PATH)),
        []
    )
    return (documents,)


@app.cell
def _(documents):
    documents[1]
    return


app._unparsable_cell(
    r"""
    #with open(OUTPUT_JSON_PATH, \"w\", encoding=\"utf-8\") as f:
        json.dump(documents, f, ensure_ascii=False)
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
