#!/usr/bin/env python3
"""
Migrate Pinecone index from cosine to dotproduct metric.

Recreates the index with dotproduct metric to support hybrid dense+sparse queries,
then re-upserts all recipe vectors with dense + sparse embeddings.

Dry-run by default. Pass --execute to actually perform the migration.

Usage:
    python scripts/migrate_index_to_dotproduct.py            # dry-run
    python scripts/migrate_index_to_dotproduct.py --execute   # perform migration
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.combined_recipes import RECIPE_RECORDS  # noqa: E402
from utils.embedding_helper import embed_records  # noqa: E402
from utils.sparse_helper import build_sparse_encoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate Pinecone index from cosine to dotproduct metric."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration. Without this flag, only a dry-run is performed.",
    )
    return parser.parse_args()


def upsert_vectors(index, namespace, prepared, sparse_encoder):
    """Upsert embedded vectors into Pinecone (mirrors main.py logic)."""
    vectors = []
    for record_id, dense_values, metadata in prepared:
        sparse_values = sparse_encoder.encode(metadata.get("content", ""))
        vector = {
            "id": record_id,
            "values": dense_values,
            "metadata": metadata,
        }
        if sparse_values["indices"]:
            vector["sparse_values"] = sparse_values
        vectors.append(vector)
    print(f"Upserting {len(vectors)} vectors into namespace '{namespace}'...")
    index.upsert(namespace=namespace, vectors=vectors)
    print(f"Successfully upserted {len(vectors)} vectors.")


def main():
    load_dotenv()

    args = parse_args()
    dry_run = not args.execute

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not set in .env")
        sys.exit(1)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    index_name = os.getenv("PINECONE_INDEX", "recipes-vector-db")
    namespace = os.getenv("PINECONE_NAMESPACE", "main_recipes")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    sparse_hash_dim = int(os.getenv("SPARSE_HASH_DIM", str(2**18)))
    sparse_min_doc_freq = int(
        os.getenv("SPARSE_MIN_DOC_FREQ", os.getenv("SPARSE_MIN_DF", "1"))
    )

    pc = Pinecone(api_key=api_key)

    # Describe current index
    try:
        desc = pc.describe_index(index_name)
    except Exception as e:
        print(f"Error: Could not describe index '{index_name}': {e}")
        sys.exit(1)

    current_metric = desc.metric
    dimension = desc.dimension

    print(f"Index: {index_name}")
    print(f"Current metric: {current_metric}")
    print(f"Dimension: {dimension}")

    if current_metric == "dotproduct":
        print("Index is already using dotproduct metric. No migration needed.")
        sys.exit(0)

    if current_metric != "cosine":
        print(
            f"Warning: Expected metric 'cosine' but found '{current_metric}'. "
            "Proceeding anyway."
        )

    target_metric = "dotproduct"
    cloud = "aws"
    region = "us-east-1"

    print(f"\nMigration plan:")
    print(f"  1. Delete index '{index_name}'")
    print(f"  2. Create index '{index_name}' with metric='{target_metric}', "
          f"dimension={dimension}, cloud={cloud}, region={region}")
    print(f"  3. Embed and upsert {len(RECIPE_RECORDS)} recipes "
          f"(dense + sparse) into namespace '{namespace}'")

    if dry_run:
        print("\n[DRY RUN] No changes made. Pass --execute to perform migration.")
        return

    # Step 1: Delete the old index
    print(f"\nDeleting index '{index_name}'...")
    pc.delete_index(index_name)
    print("Index deleted.")

    # Wait briefly for deletion to propagate
    print("Waiting for deletion to propagate...")
    time.sleep(5)

    # Step 2: Create new index with dotproduct
    print(f"Creating index '{index_name}' with metric='{target_metric}'...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=target_metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )

    # Wait for index to be ready
    print("Waiting for index to be ready...")
    while not pc.describe_index(index_name).status.get("ready", False):
        time.sleep(2)
    print("Index is ready.")

    # Step 3: Embed and upsert all recipes
    index = pc.Index(index_name)
    openai_client = OpenAI(api_key=openai_api_key)

    print(f"\nEmbedding {len(RECIPE_RECORDS)} recipes...")
    prepared = embed_records(RECIPE_RECORDS, embedding_model, openai_client)

    sparse_encoder = build_sparse_encoder(
        RECIPE_RECORDS, dim=sparse_hash_dim, min_df=sparse_min_doc_freq
    )
    upsert_vectors(index, namespace, prepared, sparse_encoder)

    print("Waiting 10 seconds for vectors to become queryable...")
    time.sleep(10)

    print(f"\nMigration complete. Index '{index_name}' now uses '{target_metric}' metric.")


if __name__ == "__main__":
    main()
