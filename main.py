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

    # Only use dense search (bypass sparse search)
    python main.py --dense-only
"""

import argparse
import logging
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
from validation.detector import PromptInjectionDetector
from validation.sanitizer import InputSanitizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt injection validation
# ---------------------------------------------------------------------------

_detector = PromptInjectionDetector()
_sanitizer = InputSanitizer(max_input_length=2000)
BLOCK_SCORE = int(os.environ.get("PROMPT_INJECTION_BLOCK_SCORE", "10"))


class PromptInjectionError(Exception):
    """Raised when input scores at or above the block threshold."""

    def __init__(self, risk_result: dict):
        self.risk_result = risk_result
        super().__init__(
            f"Prompt injection blocked: risk={risk_result['risk_level']}, "
            f"score={risk_result['score']}"
        )


def validate_and_sanitize_text(text: str) -> str:
    """Validate input for prompt injection and return sanitized text.

    Raises PromptInjectionError if the risk score meets or exceeds BLOCK_SCORE.
    """
    risk = _detector.calculate_risk_score(text, block_score=BLOCK_SCORE)
    if risk["should_block"]:
        raise PromptInjectionError(risk)
    return _sanitizer.sanitize(text)


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
    dense_only: bool = False,
    diagnostics: dict | None = None,
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
        dense_only: If True, skip sparse search entirely (useful for benchmarking)

    Returns:
        Formatted response string
    """
    # Validate and sanitize user input before any LLM/search calls
    user_query = validate_and_sanitize_text(user_query)

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "mode": "dense_only" if dense_only else "hybrid",
                "params": {
                    "threshold": threshold,
                    "sparse_threshold": sparse_threshold,
                    "min_dense_hits": min_dense_hits,
                    "dense_top_k": dense_top_k,
                    "sparse_top_k": sparse_top_k,
                },
                "route": None,
            }
        )

    # Embed the query
    query_vector = embed_text(user_query, embedding_model, openai_client)

    # Search the RAG database (dense first)
    dense_hits = search_dense_recipes(
        index, namespace, query_vector, user_query, top_k=dense_top_k
    )

    dense_passes = False
    dense_top_score = 0.0
    dense_sorted_hits: list[dict] = []
    if dense_hits:
        dense_passes, dense_top_score, dense_sorted_hits = check_score_threshold(
            dense_hits, threshold
        )
    if diagnostics is not None:
        diagnostics["dense"] = {
            "hit_count": len(dense_hits),
            "passes_threshold": dense_passes,
            "top_score": dense_top_score,
            "top_ids": [hit.get("_id") for hit in dense_sorted_hits[:3]],
        }

    sparse_hits = []
    sparse_passes = False
    sparse_top_score = 0.0
    sparse_sorted_hits: list[dict] = []
    if not dense_only:
        sparse_hits = search_sparse_recipes(
            index, namespace, query_vector, sparse_encoder, user_query, top_k=sparse_top_k
        )
        if sparse_hits:
            sparse_passes, sparse_top_score, sparse_sorted_hits = check_score_threshold(
                sparse_hits, sparse_threshold
            )
    if diagnostics is not None:
        diagnostics["sparse"] = {
            "hit_count": len(sparse_hits),
            "passes_threshold": sparse_passes,
            "top_score": sparse_top_score,
            "top_ids": [hit.get("_id") for hit in sparse_sorted_hits[:3]],
        }

    dense_eligible = dense_passes and (dense_only or len(dense_hits) >= min_dense_hits)

    # Hybrid mode: evaluate dense and sparse, then choose the stronger result.
    if dense_eligible and (dense_only or dense_top_score >= sparse_top_score):
        response = generate_recipe_response(user_query, dense_sorted_hits, openai_client)
        if diagnostics is not None:
            diagnostics["route"] = "dense_accepted"
        return format_response(response, RecipeSource.RAG_DATABASE, dense_top_score)

    if sparse_passes:
        response = generate_recipe_response(user_query, sparse_sorted_hits, openai_client)
        if diagnostics is not None:
            diagnostics["route"] = "sparse_accepted"
        return format_response(response, RecipeSource.RAG_SPARSE, sparse_top_score)

    if dense_hits:
        response = generate_recipe_response(user_query, dense_sorted_hits, openai_client)
        if diagnostics is not None:
            diagnostics["route"] = "dense_fallback"
        return format_response(response, RecipeSource.RAG_DATABASE, dense_top_score)

    if sparse_hits:
        response = generate_recipe_response(user_query, sparse_sorted_hits, openai_client)
        if diagnostics is not None:
            diagnostics["route"] = "sparse_fallback"
        return format_response(response, RecipeSource.RAG_SPARSE, sparse_top_score)

    # No good match - generate a recipe
    response = generate_fallback_recipe(user_query, openai_client)
    if diagnostics is not None:
        diagnostics["route"] = "llm_fallback"
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
    dense_only: bool = False,
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
                dense_only=dense_only,
            )
            print(result)
        except PromptInjectionError:
            print(
                "\nSorry, your query could not be processed because it "
                "contains content that resembles a prompt injection attempt. "
                "Please rephrase your question to focus on finding a recipe.\n"
            )
        except Exception as e:
            print(format_error(str(e)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    threshold_default = float(os.getenv("MATCH_THRESHOLD", "0.10"))
    sparse_threshold_default = float(os.getenv("SPARSE_THRESHOLD", "0.0"))
    min_dense_hits_default = int(os.getenv("MIN_DENSE_HITS", "3"))
    dense_top_k_default = int(os.getenv("DENSE_TOP_K", "10"))
    sparse_top_k_default = int(os.getenv("SPARSE_TOP_K", "10"))

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
  python main.py --dense-only                 # Only use dense search (bypass sparse)
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
        default=threshold_default,
        help="Minimum relevance score to consider a match "
        f"(default: {threshold_default}).",
    )
    parser.add_argument(
        "--sparse-threshold",
        type=float,
        default=sparse_threshold_default,
        help="Minimum relevance score for sparse fallback "
        f"(default: {sparse_threshold_default}).",
    )
    parser.add_argument(
        "--min-dense-hits",
        type=int,
        default=min_dense_hits_default,
        help="Minimum dense hits required before skipping sparse fallback "
        f"(default: {min_dense_hits_default}).",
    )
    parser.add_argument(
        "--dense-top-k",
        type=int,
        default=dense_top_k_default,
        help=f"Number of dense results to retrieve (default: {dense_top_k_default}).",
    )
    parser.add_argument(
        "--sparse-top-k",
        type=int,
        default=sparse_top_k_default,
        help=f"Number of sparse results to retrieve (default: {sparse_top_k_default}).",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Only use dense search and bypass sparse search (useful for benchmarking).",
    )
    return parser.parse_args(argv)


def main() -> None:
    load_dotenv()

    sparse_hash_dim = int(os.getenv("SPARSE_HASH_DIM", str(2**18)))
    sparse_min_doc_freq_value = os.getenv(
        "SPARSE_MIN_DOC_FREQ", os.getenv("SPARSE_MIN_DF", "1")
    )
    sparse_min_doc_freq = int(sparse_min_doc_freq_value)

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
    if args.dense_only:
        print("Dense-only mode: sparse search is disabled.")

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
        try:
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
                dense_only=args.dense_only,
            )
            print(result)
        except PromptInjectionError:
            print(
                "Error: Your query could not be processed because it "
                "contains content that resembles a prompt injection attempt. "
                "Please rephrase your question to focus on finding a recipe."
            )
            sys.exit(1)
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
            dense_only=args.dense_only,
        )


if __name__ == "__main__":
    main()
