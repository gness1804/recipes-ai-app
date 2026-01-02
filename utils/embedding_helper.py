from openai import OpenAI  # type: ignore[import]


def embed_records(
    records: list[dict[str, str]], model: str, client: OpenAI
) -> list[tuple[str, list[float], dict]]:
    """Return [(id, vector, metadata)] ready for Pinecone upsert."""
    contents = [record["content"] for record in records]
    response = client.embeddings.create(model=model, input=contents)
    if not response.data:
        raise ValueError("Error: Failed to embed records.")
    return [
        (
            record["_id"],
            data.embedding,
            {k: v for k, v in record.items() if k != "_id"},
        )
        for record, data in zip(records, response.data)
    ]


def embed_text(text: str, model: str, client: OpenAI) -> list[float]:
    """Embed a single string and return the vector."""
    response = client.embeddings.create(model=model, input=text)
    if not response.data:
        raise ValueError("Error: Failed to embed text.")
    return response.data[0].embedding