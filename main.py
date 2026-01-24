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

from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone

from utils.embedding_helper import embed_records, embed_text
from utils.llm_helper import (
    generate_recipe_response,
    generate_fallback_recipe,
    check_score_threshold,
)
from utils.response_formatter import (
    format_response,
    format_error,
    format_welcome,
    format_prompt,
    RecipeSource,
)
from data.combined_recipes import RECIPE_RECORDS


def upsert_vectors(
    index, namespace: str, prepared: Iterable[tuple[str, list[float], dict]]
) -> None:
    """Upsert embedded vectors into Pinecone."""
    vectors = list(prepared)
    print(f"Upserting {len(vectors)} vectors into namespace '{namespace}'...")
    index.upsert(namespace=namespace, vectors=vectors)
    print(f"Successfully upserted {len(vectors)} vectors.")


def search_recipes(
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


def process_query(
    user_query: str,
    index,
    namespace: str,
    openai_client: OpenAI,
    embedding_model: str,
    threshold: float,
) -> str:
    """
    Process a user query: search RAG database, evaluate match, respond or fallback.

    Args:
        user_query: The user's recipe question
        index: Pinecone index
        namespace: Pinecone namespace
        openai_client: OpenAI client
        embedding_model: Model for embeddings
        threshold: Minimum score threshold for a "good" match

    Returns:
        Formatted response string
    """
    # Embed the query
    query_vector = embed_text(user_query, embedding_model, openai_client)

    # Search the RAG database
    hits = search_recipes(index, namespace, query_vector, user_query)

    if not hits:
        # No results at all - generate a recipe
        response = generate_fallback_recipe(user_query, openai_client)
        return format_response(response, RecipeSource.LLM_GENERATED)

    # Check if the top result meets our threshold
    passes_threshold, top_score, sorted_hits = check_score_threshold(hits, threshold)

    if passes_threshold:
        # Good match found - use RAG results
        response = generate_recipe_response(user_query, sorted_hits, openai_client)
        return format_response(response, RecipeSource.RAG_DATABASE, top_score)
    else:
        # No good match - generate a recipe
        print(f"No match above threshold ({threshold}). Best score: {top_score:.2f}")
        response = generate_fallback_recipe(user_query, openai_client)
        return format_response(response, RecipeSource.LLM_GENERATED)


def run_interactive_mode(
    index,
    namespace: str,
    openai_client: OpenAI,
    embedding_model: str,
    threshold: float,
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
                threshold,
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
        """,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to process (skips interactive mode).",
    )
    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Upsert recipe records to Pinecone before querying.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.7,
        help="Minimum relevance score to consider a match (default: 0.7).",
    )
    return parser.parse_args()


def main() -> None:
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

    # Handle upsert if requested
    if args.upsert:
        print(f"\nPreparing to upsert {len(RECIPE_RECORDS)} recipes...")
        prepared = embed_records(RECIPE_RECORDS, embedding_model, openai_client)
        upsert_vectors(index, namespace, prepared)
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
            args.threshold,
        )
        print(result)
    else:
        # Interactive mode
        run_interactive_mode(
            index,
            namespace,
            openai_client,
            embedding_model,
            args.threshold,
        )


if __name__ == "__main__":
    main()
