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
    return (json,)


@app.cell
def _():
    from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    from langchain_core.documents.base import Document
    return


@app.cell
def _():
    import redis
    import uuid
    return redis, uuid


@app.cell
def _(json):
    dataset_filepath = "./documents_with_visual_analysis.json"
    with open(dataset_filepath, "r") as file:
        pages = json.load(file)
    return (pages,)


@app.cell
def _(pages):
    [p for p in pages.items()][252]
    #pages["doc_000_page_0001"]
    return


@app.cell
def _():
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    #md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("**", "Heavy")])
    return


@app.cell
def _():
    #def split_page(page_content: str):
    #    md_split_text = md_splitter.split_text(page_content)
    #    split_text = text_splitter.split_documents(md_split_text)
    #    return split_text
    return


@app.cell
def _():
    #def clean_markdown(text: str):
    #    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE) # remove headings
    #    text = re.sub(r"(\*\*|\*|__|_)(.*?)\1", r"\2", text) # remove italic and bold
    #    text = re.sub(r"^\s*(\(?\d+[a-z]?[.)]|\(?[a-zA-Z][.)])\s+", "", text, flags=re.MULTILINE) # remove "points"
    #    return text.strip()
    return


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
def _(pages, pipeline, uuid):
    for page in pages.values():
        json_doc = {
            "document_filename": page["document_filename"],
            "page_number": page["page_index"],
            "description": page["visual_analysis"]["analysis"],
            "raw_content": page["content"]
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
    client.json().get("doc:02c8bbb1-5977-4570-8e61-ca2b1bc0ae36")
    return


@app.cell
def _(client, keys):
    descriptions = client.json().mget(keys, "$.description")
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
    client.json().get("doc:02c8bbb1-5977-4570-8e61-ca2b1bc0ae36")
    return


@app.cell
def _():
    from redis.commands.search.field import (
        TextField,
        VectorField,
        NumericField,
        TagField
    )
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    return (
        IndexDefinition,
        IndexType,
        NumericField,
        Query,
        TagField,
        TextField,
        VectorField,
    )


@app.cell
def _(embeddings):
    VECTOR_DIMENSION = len(embeddings[0])
    return (VECTOR_DIMENSION,)


@app.cell
def _(
    IndexDefinition,
    IndexType,
    NumericField,
    TagField,
    TextField,
    VECTOR_DIMENSION,
    VectorField,
    client,
):
    schema = (
        TagField("$.document_filename", as_name="document_filename"),
        NumericField("$.page_number", as_name="page_number"),
        TextField("$.description", as_name="chunk_content"),
        TextField("$.raw_content", as_name="raw_content"),
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
    encoded_query = embedder.embed_query("What is the detailed energetic agents for the future")
    return (encoded_query,)


@app.cell
def _(Query):
    query = (
        Query('(*)=>[KNN 3 @vector $query_vector AS vector_score]')
         .sort_by('vector_score')
         .return_fields('vector_score', 'chunk_content', 'raw_content')
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


@app.cell
def _(Query):
    another_query = (
        Query('(*)=>[KNN 5 @vector $query_vector AS vector_distance]')
         .sort_by("vector_distance")
         .return_fields("vector", "vector_distance", "chunk_content")
         .dialect(3)
    )
    # *=>[KNN 5 @embedding $vector AS vector_distance] RETURN 3 chunk_content vector vector_distance SORTBY vector_distance ASC DIALECT 2 LIMIT 0 5 <class 'redisvl.query.query.VectorQuery'>
    return (another_query,)


@app.cell
def _(another_query, client, encoded_query, np):
    client.ft('idx:doc_vss').search(
        another_query,
        {
          'query_vector': np.array(encoded_query, dtype=np.float32).tobytes()
        }
    ).docs
    return


if __name__ == "__main__":
    app.run()
