"""
Response formatter for recipe chatbot CLI output.

Handles formatting recipe responses for display, including source attribution.
"""

from enum import Enum


class RecipeSource(Enum):
    """Source of the recipe response."""
    RAG_DATABASE = "rag"
    RAG_SPARSE = "rag_sparse"
    LLM_GENERATED = "generated"


def format_response(
    response_text: str,
    source: RecipeSource,
    score: float | None = None,
) -> str:
    """
    Format a recipe response for CLI display with source attribution.

    Args:
        response_text: The recipe response text from LLM
        source: Where the recipe came from (RAG database or generated)
        score: Relevance score if from RAG database

    Returns:
        Formatted string for CLI output
    """
    separator = "=" * 60

    header = _build_header(source, score)

    output_parts = [
        separator,
        header,
        separator,
        "",
        response_text,
        "",
        separator,
    ]

    return "\n".join(output_parts)


def _build_header(source: RecipeSource, score: float | None) -> str:
    """Build the header line based on source type."""
    if source == RecipeSource.RAG_DATABASE:
        score_str = f" (relevance: {score:.2f})" if score is not None else ""
        return f"Recipe from your collection{score_str}"
    if source == RecipeSource.RAG_SPARSE:
        score_str = f" (relevance: {score:.2f})" if score is not None else ""
        return f"Recipe from your collection (sparse search){score_str}"
    else:
        return "Generated recipe (no match found in your collection)"


def format_error(error_message: str) -> str:
    """
    Format an error message for CLI display.

    Args:
        error_message: The error description

    Returns:
        Formatted error string
    """
    separator = "=" * 60
    return f"\n{separator}\nError: {error_message}\n{separator}\n"


def format_welcome() -> str:
    """Return the welcome message for interactive mode."""
    return """
================================================================================
                         Recipe Chatbot (MVP)
================================================================================

Ask me about recipes! I'll search your personal recipe collection first.
If no good match is found, I'll generate a recipe for you.

Examples:
  - "Give me a good seafood recipe for a weeknight"
  - "Easy chicken dinner under 30 minutes"
  - "Vegetarian soup recipe"

Type 'quit' or 'exit' to stop.
================================================================================
"""


def format_prompt() -> str:
    """Return the input prompt for interactive mode."""
    return "\nYour question: "
