import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import json
    import re
    from functools import reduce
    return json, re, reduce


@app.cell
def _():
    from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    from langchain_core.documents.base import Document
    return Document, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


@app.cell
def _():
    import redis
    import uuid
    return redis, uuid


@app.cell
def _(json):
    dataset_filepath = "./infered_pages_latest.json"
    with open(dataset_filepath, "r") as file:
        infered_pages = json.load(file)
    return (infered_pages,)


@app.cell
def _(infered_pages):
    infered_pages
    return


@app.cell
def _(MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("**", "Heavy")])
    return md_splitter, text_splitter


@app.cell
def _(md_splitter, text_splitter):
    def split_page(page_content: str):
        md_split_text = md_splitter.split_text(page_content)
        split_text = text_splitter.split_documents(md_split_text)
        return split_text
    return (split_page,)


@app.cell
def _(re):
    def clean_markdown(text: str):
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE) # remove headings
        text = re.sub(r"(\*\*|\*|__|_)(.*?)\1", r"\2", text) # remove italic and bold
        text = re.sub(r"^\s*(\(?\d+[a-z]?[.)]|\(?[a-zA-Z][.)])\s+", "", text, flags=re.MULTILINE) # remove "points"
        return text.strip()
    return (clean_markdown,)


@app.cell
def _(Document, clean_markdown, infered_pages, reduce, split_page):
    paired_pages_documents = reduce(
        lambda res, split_chunks: [
            *res,
            *[
                Document(page_content=clean_markdown(p1.page_content) + "\n\n" +                        clean_markdown(p2.page_content))
                for p1, p2 in zip(split_chunks[:-1], split_chunks[1:])
            ]
        ],
        map(lambda v: split_page(v[0]), infered_pages.values()),
        []
    )
    paired_pages_documents
    return (paired_pages_documents,)


@app.cell
def _():
    from langchain_ollama import OllamaEmbeddings
    return (OllamaEmbeddings,)


@app.cell
def _(OllamaEmbeddings):
    embedder = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    return (embedder,)


@app.cell
def _(redis):
    client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    pipeline = client.pipeline()
    return client, pipeline


@app.cell
def _(paired_pages_documents, pipeline, uuid):
    for doc in map(lambda d: d.page_content, paired_pages_documents):
        json_doc = {
            "chunk_content": doc
        }
        redis_key = f"doc:{uuid.uuid4()}"
        pipeline.json().set(redis_key, "$", json_doc)
    res = pipeline.execute()
    return


@app.cell
def _(client):
    keys = sorted(client.keys("doc:*"))
    keys
    return (keys,)


@app.cell
def _(client):
    client.json().get("doc:0256aecb-0dde-4ecc-b4ac-9f23c6a25bdc")
    return


@app.cell
def _(client, keys):
    descriptions = client.json().mget(keys, "$.chunk_content")
    descriptions = [item for sublist in descriptions for item in sublist]
    descriptions
    return (descriptions,)


@app.cell
def _(descriptions, embedder):
    embeddings = embedder.embed_documents(descriptions)
    return (embeddings,)


@app.cell
def _(embeddings):
    embeddings[0:2]
    return


@app.cell
def _(embeddings, keys, pipeline):
    for key, embedding in zip(keys, embeddings):
        pipeline.json().set(key, "$.embedding", embedding)
    pipeline.execute()
    return


@app.cell
def _(client):
    client.json().get("doc:00ee380f-2852-4ff0-bf4c-f36cf01791a6")
    return


@app.cell
def _():
    from redis.commands.search.field import (
        TextField,
        VectorField,
    )
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    return IndexDefinition, IndexType, Query, TextField, VectorField


@app.cell
def _(embeddings):
    VECTOR_DIMENSION = len(embeddings[0])
    return (VECTOR_DIMENSION,)


@app.cell
def _(
    IndexDefinition,
    IndexType,
    TextField,
    VECTOR_DIMENSION,
    VectorField,
    client,
):
    schema = (
        TextField("$.chunk_content", as_name="chunk_content"),
        VectorField(
            "$.embedding",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIMENSION,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="vector",
        ),
    )
    definition = IndexDefinition(prefix=["doc:"], index_type=IndexType.JSON)
    client.ft("idx:doc_vss").create_index(fields=schema, definition=definition)
    return


@app.cell
def _(client):
    info = client.ft("idx:doc_vss").info()
    num_docs = info["num_docs"]
    indexing_failures = info["hash_indexing_failures"]
    return indexing_failures, num_docs


@app.cell
def _(indexing_failures, num_docs):
    num_docs, indexing_failures
    return


@app.cell
def _(embedder):
    encoded_query = embedder.embed_query("What is the available biomass in Sion ?")
    return (encoded_query,)


@app.cell
def _(Query):
    query = (
        Query('(*)=>[KNN 3 @vector $query_vector AS vector_score]')
         .sort_by('vector_score')
         .return_fields('vector_score', 'chunk_content')
         .dialect(2)
    )
    return (query,)


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(client, encoded_query, np, query):
    client.ft('idx:doc_vss').search(
        query,
        {
          'query_vector': np.array(encoded_query, dtype=np.float32).tobytes()
        }
    ).docs
    return


if __name__ == "__main__":
    app.run()
