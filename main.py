import argparse
import os
import time
from typing import Iterable

from openai import OpenAI  # type: ignore[import]
from dotenv import load_dotenv
from pinecone import Pinecone  # type: ignore[import]
from utils.embedding_helper import embed_records, embed_text
from data.seeds import SEED_RECORDS


def upsert_vectors(index, namespace: str, prepared: Iterable[tuple[str, list[float], dict]]) -> None:
    vectors = list(prepared)
    print(f"Upserting {len(vectors)} vectors into {namespace}")
    index.upsert(namespace=namespace, vectors=vectors)


def search_recipes(
    index, namespace: str, user_query: str, query_vector: list[float], top_k: int = 5
) -> None:
    """Run a reranked search to demonstrate recommendations."""
    reranked = index.search(
        namespace=namespace,
        query={
            "top_k": top_k,
            "vector": {"values": query_vector},
        },
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": 10,
            "rank_fields": ["content"],
            "query": user_query,
        },
    )

    hits = reranked["result"]["hits"]
    if not hits:
        print(f"No Pinecone hits for '{user_query}', falling back to the next data source (not implemented).")
        return

    print(f"Top {len(hits)} hits for '{user_query}':")
    for hit in hits:
        fields = hit["fields"]
        print(
            f"- {hit['_id']} ({round(hit['_score'], 2)}): {fields['content'][:80]}… | "
            f"category={fields.get('category')} prep_time={fields.get('prep_time')} rating={fields.get('rating')}"
        )


def run_sample_queries(
    index, namespace: str, client: OpenAI, embedding_model: str
) -> None:
    """Execute a few queries to verify the recommendation path."""
    queries = [
        "Give me a good seafood recipe that works on a weeknight.",
        # "Quick vegetarian dinner with tofu",
        # "Gluten-free sheet pan idea",
        # "Highest rated recipe",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        query_vector = embed_text(query, embedding_model, client)
        search_recipes(index, namespace, query, query_vector)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed Pinecone recipes and run a few sample queries."
    )
    parser.add_argument(
        "--skip-upsert",
        action="store_true",
        help="Skip the upsert step (useful after initial seed data load).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Error: Please set PINECONE_API_KEY in your .env file.")

    pc = Pinecone(api_key=api_key, environment=os.getenv("PINECONE_ENVIRONMENT"))
    if not pc:
        raise ValueError("Error: Failed to connect to Pinecone.")

    index_name = os.getenv("PINECONE_INDEX", "recipes-vector-db")
    if not index_name:
        raise ValueError("Error: Please set PINECONE_INDEX in your .env file.")

    index = pc.Index(index_name)
    namespace = os.getenv("PINECONE_NAMESPACE", "main_recipes")

    args = parse_args()
    print(f"Connected to Pinecone index: {index_name} (namespace={namespace})")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    if not embedding_model:
        raise ValueError("Error: Please set EMBEDDING_MODEL in your .env file.")

    if not args.skip_upsert:
        prepared = embed_records(SEED_RECORDS, embedding_model, client)
        upsert_vectors(index, namespace, prepared)
        print("Waiting 10 seconds before querying to allow vectors to become queryable.")
        time.sleep(10)

    run_sample_queries(index, namespace, client, embedding_model)


if __name__ == "__main__":
    main()