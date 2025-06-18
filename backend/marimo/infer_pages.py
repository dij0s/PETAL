import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This notebook uses ColPali-v1.3 to embed and score PDF page images and their extracted text content for downstream RAG applications.

    The input is produced by `extract_content.py` and consists of a JSON file with documents, each containing pages with:
    - `"pixmap_render"`: base64 PNG image (data URI)
    - `"content"`: extracted text

    The script computes similarity scores between each page's image and its text content.

    Requirements:
    - `colpali_engine`
    - `torch`
    - `Pillow`
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import json
    import base64
    import io
    from PIL import Image
    import torch
    from typing import List
    from colpali_engine.models import ColPali, ColPaliProcessor
    return json, base64, io, Image, torch, List, ColPali, ColPaliProcessor


@app.cell
def _():
    EXTRACTED_JSON_PATH = "./extracted_pages.json"
    OUTPUT_JSON_PATH = "./colpali_scores.json"
    MODEL_NAME = "vidore/colpali-v1.3"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    return EXTRACTED_JSON_PATH, OUTPUT_JSON_PATH, MODEL_NAME, DEVICE


@app.cell
def _(base64, io, Image):
    def decode_base64_image(data_uri: str) -> Image.Image:
        header, encoded = data_uri.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(image_bytes))
    return decode_base64_image,


@app.cell
def _(ColPali, ColPaliProcessor, MODEL_NAME, DEVICE, torch):
    model = ColPali.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    ).eval()
    processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
    return model, processor


@app.cell
def _(EXTRACTED_JSON_PATH, json, decode_base64_image):
    with open(EXTRACTED_JSON_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    images = []
    queries = []
    page_ids = []

    for doc in documents:
        for page in doc["pages"]:
            img = decode_base64_image(page["pixmap_render"])
            images.append(img)
            queries.append(page["content"])
            page_ids.append(f"{doc['filename']}_page_{page['page_number']}")

    return images, queries, page_ids


@app.cell
def _(images, queries, model, processor, torch):
    # Batch process all images and queries
    batch_images = processor.process_images(images).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    return scores


@app.cell
def _(scores, page_ids, json, OUTPUT_JSON_PATH):
    results = {pid: float(score) for pid, score in zip(page_ids, scores.tolist())}
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved scores to {OUTPUT_JSON_PATH}")
    return results


if __name__ == "__main__":
    app.run()
