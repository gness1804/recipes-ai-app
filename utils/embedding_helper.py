from openai import OpenAI  # type: ignore[import]


def embed_records(
    records: list[dict], model: str, client: OpenAI
) -> list[tuple[str, list[float], dict]]:
    """
    Return [(id, vector, metadata)] ready for Pinecone upsert.

    Flattens nested metadata structure into top-level fields, as Pinecone
    requires flat metadata (strings, numbers, booleans, or lists of strings).
    Filters out None values and empty arrays which Pinecone doesn't accept.
    """
    contents = [record["content"] for record in records]
    response = client.embeddings.create(model=model, input=contents)
    if not response.data:
        raise ValueError("Error: Failed to embed records.")

    result = []
    for record, data in zip(records, response.data):
        # Start with content
        flat_metadata = {"content": record["content"]}

        # Flatten nested metadata into top level
        if "metadata" in record:
            for key, value in record["metadata"].items():
                # Skip None values and empty arrays - Pinecone doesn't accept them
                if value is None:
                    continue
                if isinstance(value, list) and len(value) == 0:
                    continue
                flat_metadata[key] = value

        result.append((record["_id"], data.embedding, flat_metadata))

    return result


def embed_text(text: str, model: str, client: OpenAI) -> list[float]:
    """Embed a single string and return the vector."""
    response = client.embeddings.create(model=model, input=text)
    if not response.data:
        raise ValueError("Error: Failed to embed text.")
    return response.data[0].embedding