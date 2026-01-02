from openai import OpenAI

def embed_records(records: list[dict[str, str]], model: str, client: OpenAI) -> list[tuple[str, list[float], dict]]:
    """Return [(id, vector, metadata)] ready for Pinecone upsert."""
    contents = [record["content"] for record in records]
    response = client.embeddings.create(model=model, input=contents)
    if not response.data:
        raise ValueError("Error: Failed to embed records.")
    return [
        (
            record["_id"],
            data.embedding,
            {k: v for k, v in record.items() if k not in {"_id", "content"}},
        )
        for record, data in zip(records, response.data)
    ]