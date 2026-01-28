#!/usr/bin/env python3
"""
Recipe Chatbot MVP

A RAG-based recipe assistant that searches your personal recipe collection
and falls back to LLM generation when no suitable match is found.

Usage:
    # Single query mode
    python main.py --query "Give me a good seafood recipe"

    # Interactive mode (default)
    python main.py

    # Upsert recipes to Pinecone first
    python main.py --upsert

    # Adjust match threshold
    python main.py --threshold 0.6
"""

import argparse
import os
import sys
import time
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from data.combined_recipes import RECIPE_RECORDS
from utils.embedding_helper import embed_records, embed_text
from utils.llm_helper import (
    check_score_threshold,
    generate_fallback_recipe,
    generate_recipe_response,
)
from utils.response_formatter import (
    RecipeSource,
    format_error,
    format_prompt,
    format_response,
    format_welcome,
)
from utils.sparse_helper import SparseEncoder, build_sparse_encoder


def upsert_vectors(
    index,
    namespace: str,
    prepared: Iterable[tuple[str, list[float], dict]],
    sparse_encoder: SparseEncoder,
) -> None:
    """Upsert embedded vectors into Pinecone."""
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


def search_dense_recipes(
    index, namespace: str, query_vector: list[float], user_query: str, top_k: int = 10
) -> list[dict]:
    """
    Search for recipes using vector similarity and reranking.

    Args:
        index: Pinecone index
        namespace: Pinecone namespace
        query_vector: Embedded query vector
        user_query: Original user query text (for reranking)
        top_k: Number of results to return

    Returns:
        List of hit dictionaries with _id, _score, and fields
    """
    reranked = index.search(
        namespace=namespace,
        query={
            "top_k": top_k,
            "vector": {"values": query_vector},
        },
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": top_k,
            "rank_fields": ["content"],
            "query": user_query,
        },
    )

    return reranked.get("result", {}).get("hits", [])


def search_sparse_recipes(
    index,
    namespace: str,
    query_vector: list[float],
    sparse_encoder: SparseEncoder,
    user_query: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Search for recipes using sparse (lexical) similarity and reranking.

    Args:
        index: Pinecone index
        namespace: Pinecone namespace
        query_vector: Dense embedding vector (required by Pinecone for dense indexes)
        sparse_encoder: Encoder used for sparse vectors
        user_query: Original user query text (for reranking)
        top_k: Number of results to return

    Returns:
        List of hit dictionaries with _id, _score, and fields
    """
    sparse_values = sparse_encoder.encode(user_query)
    if not sparse_values["indices"]:
        return []

    reranked = index.search(
        namespace=namespace,
        query={
            "top_k": top_k,
            "vector": {
                "values": query_vector,
                "sparse_indices": sparse_values["indices"],
                "sparse_values": sparse_values["values"],
            },
        },
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": top_k,
            "rank_fields": ["content"],
            "query": user_query,
        },
    )

    return reranked.get("result", {}).get("hits", [])


def process_query(
    user_query: str,
    index,
    namespace: str,
    openai_client: OpenAI,
    embedding_model: str,
    sparse_encoder: SparseEncoder,
    threshold: float,
    sparse_threshold: float,
    min_dense_hits: int,
    dense_top_k: int,
    sparse_top_k: int,
) -> str:
    """
    Process a user query: search RAG database, evaluate match, respond or fallback.

    Args:
        user_query: The user's recipe question
        index: Pinecone index
        namespace: Pinecone namespace
        openai_client: OpenAI client
        embedding_model: Model for embeddings
        sparse_encoder: Encoder used for sparse vectors
        threshold: Minimum score threshold for a "good" match
        sparse_threshold: Minimum score threshold for sparse fallback
        min_dense_hits: Minimum dense hit count before skipping sparse fallback
        dense_top_k: Dense search top_k
        sparse_top_k: Sparse search top_k

    Returns:
        Formatted response string
    """
    # Embed the query
    query_vector = embed_text(user_query, embedding_model, openai_client)

    # Search the RAG database (dense first)
    dense_hits = search_dense_recipes(
        index, namespace, query_vector, user_query, top_k=dense_top_k
    )

    dense_fallback = None
    if dense_hits:
        passes_threshold, top_score, sorted_hits = check_score_threshold(
            dense_hits, threshold
        )
        if passes_threshold:
            if len(dense_hits) >= min_dense_hits:
                response = generate_recipe_response(
                    user_query, sorted_hits, openai_client
                )
                return format_response(response, RecipeSource.RAG_DATABASE, top_score)
            dense_fallback = (sorted_hits, top_score)

    # Fall back to sparse search for complex queries
    sparse_hits = search_sparse_recipes(
        index, namespace, query_vector, sparse_encoder, user_query, top_k=sparse_top_k
    )
    if sparse_hits:
        passes_threshold, top_score, sorted_hits = check_score_threshold(
            sparse_hits, sparse_threshold
        )
        if passes_threshold:
            response = generate_recipe_response(user_query, sorted_hits, openai_client)
            return format_response(response, RecipeSource.RAG_SPARSE, top_score)
        print(
            "Sparse search results did not meet threshold "
            f"({sparse_threshold}). Best score: {top_score:.2f}"
        )

    if dense_fallback is not None:
        sorted_hits, top_score = dense_fallback
        response = generate_recipe_response(user_query, sorted_hits, openai_client)
        return format_response(response, RecipeSource.RAG_DATABASE, top_score)

    # No good match - generate a recipe
    response = generate_fallback_recipe(user_query, openai_client)
    return format_response(response, RecipeSource.LLM_GENERATED)


def run_interactive_mode(
    index,
    namespace: str,
    openai_client: OpenAI,
    embedding_model: str,
    sparse_encoder: SparseEncoder,
    threshold: float,
    sparse_threshold: float,
    min_dense_hits: int,
    dense_top_k: int,
    sparse_top_k: int,
) -> None:
    """
    Run the chatbot in interactive mode, prompting for queries in a loop.
    """
    print(format_welcome())

    while True:
        try:
            user_input = input(format_prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        try:
            result = process_query(
                user_input,
                index,
                namespace,
                openai_client,
                embedding_model,
                sparse_encoder,
                threshold,
                sparse_threshold,
                min_dense_hits,
                dense_top_k,
                sparse_top_k,
            )
            print(result)
        except Exception as e:
            print(format_error(str(e)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recipe Chatbot MVP - Search your recipe collection or generate new recipes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Interactive mode
  python main.py --query "easy chicken dinner"  # Single query
  python main.py --upsert                     # Upsert recipes first
  python main.py --threshold 0.6              # Lower match threshold
  python main.py --dense-top-k 5              # Increase dense top-k
  python main.py --sparse-top-k 10            # Increase sparse top-k
  python main.py --sparse-threshold 0.05      # Lower sparse threshold
  python main.py --dense-top-k 5 --sparse-top-k 10 --sparse-threshold 0.05 # Increase dense top-k, sparse top-k, and lower sparse threshold
  python main.py --min-dense-hits 5           # Increase min dense hits
        """,
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Single query to process (skips interactive mode).",
    )
    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Upsert recipe records to Pinecone before querying.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.10,
        help="Minimum relevance score to consider a match (default: 0.10).",
    )
    parser.add_argument(
        "--sparse-threshold",
        type=float,
        default=0.0,
        help="Minimum relevance score for sparse fallback (default: 0.0).",
    )
    parser.add_argument(
        "--min-dense-hits",
        type=int,
        default=3,
        help="Minimum dense hits required before skipping sparse fallback (default: 3).",
    )
    parser.add_argument(
        "--dense-top-k",
        type=int,
        default=10,
        help="Number of dense results to retrieve (default: 10).",
    )
    parser.add_argument(
        "--sparse-top-k",
        type=int,
        default=10,
        help="Number of sparse results to retrieve (default: 10).",
    )
    return parser.parse_args()


def main(
    sparse_hash_dim: int,
    sparse_min_doc_freq: int,
) -> None:
    load_dotenv()

    # Validate environment
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: Please set PINECONE_API_KEY in your .env file.")
        sys.exit(1)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: Please set OPENAI_API_KEY in your .env file.")
        sys.exit(1)

    # Initialize clients
    pc = Pinecone(api_key=api_key, environment=os.getenv("PINECONE_ENVIRONMENT"))
    index_name = os.getenv("PINECONE_INDEX", "recipes-vector-db")
    index = pc.Index(index_name)
    namespace = os.getenv("PINECONE_NAMESPACE", "main_recipes")

    openai_client = OpenAI(api_key=openai_api_key)
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    args = parse_args()

    print(f"Connected to Pinecone index: {index_name} (namespace={namespace})")
    print(f"Match threshold: {args.threshold}")

    sparse_encoder = build_sparse_encoder(
        RECIPE_RECORDS,
        dim=sparse_hash_dim,
        min_df=sparse_min_doc_freq,
    )

    print(
        f"Sparse hash dim: {sparse_hash_dim} (min_doc_freq={sparse_min_doc_freq})"
    )

    # Handle upsert if requested
    if args.upsert:
        print(f"\nPreparing to upsert {len(RECIPE_RECORDS)} recipes...")
        prepared = embed_records(RECIPE_RECORDS, embedding_model, openai_client)
        upsert_vectors(index, namespace, prepared, sparse_encoder)
        print("Waiting 10 seconds for vectors to become queryable...")
        time.sleep(10)

    # Process query or enter interactive mode
    if args.query:
        # Single query mode
        result = process_query(
            args.query,
            index,
            namespace,
            openai_client,
            embedding_model,
            sparse_encoder,
            args.threshold,
            args.sparse_threshold,
            args.min_dense_hits,
            args.dense_top_k,
            args.sparse_top_k,
        )
        print(result)
    else:
        # Interactive mode
        run_interactive_mode(
            index,
            namespace,
            openai_client,
            embedding_model,
            sparse_encoder,
            args.threshold,
            args.sparse_threshold,
            args.min_dense_hits,
            args.dense_top_k,
            args.sparse_top_k,
        )


if __name__ == "__main__":
    sparse_hash_dim = int(os.getenv("SPARSE_HASH_DIM", str(2**18)))
    sparse_min_doc_freq_value = os.getenv("SPARSE_MIN_DOC_FREQ", os.getenv("SPARSE_MIN_DF", "1"))
    sparse_min_doc_freq = int(sparse_min_doc_freq_value)
    main(sparse_hash_dim, sparse_min_doc_freq)
