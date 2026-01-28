"""
Tests for the MVP recipe chatbot functionality.

Tests cover:
- Dataset combining
- Score threshold checking
- LLM response generation (mocked)
- Response formatting
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_helper import (
    check_score_threshold,
    generate_recipe_response,
    generate_fallback_recipe,
    _format_recipes_for_prompt,
)
from utils.response_formatter import (
    format_response,
    format_error,
    format_welcome,
    RecipeSource,
)
from scripts.combine_recipe_datasets import combine_datasets
from utils.embedding_helper import embed_records


class TestCombineDatasets:
    """Tests for the dataset combining functionality."""

    def test_combine_empty_datasets(self):
        result = combine_datasets([], [])
        assert result == []

    def test_combine_v2_only(self):
        v2_records = [
            {"_id": "recipe-a", "content": "A", "metadata": {}},
            {"_id": "recipe-b", "content": "B", "metadata": {}},
        ]
        result = combine_datasets(v2_records, [])
        assert len(result) == 2
        assert result[0]["_id"] == "recipe-a"
        assert result[1]["_id"] == "recipe-b"

    def test_combine_data_only(self):
        data_records = [
            {"_id": "recipe-c", "content": "C", "metadata": {}},
        ]
        result = combine_datasets([], data_records)
        assert len(result) == 1
        assert result[0]["_id"] == "recipe-c"

    def test_combine_no_duplicates(self):
        v2_records = [
            {"_id": "recipe-a", "content": "A", "metadata": {}},
        ]
        data_records = [
            {"_id": "recipe-b", "content": "B", "metadata": {}},
        ]
        result = combine_datasets(v2_records, data_records)
        assert len(result) == 2

    def test_combine_with_duplicates_v2_wins(self):
        """V2 records should override data records for duplicate IDs."""
        v2_records = [
            {"_id": "shared-recipe", "content": "V2 content", "metadata": {"source": "v2"}},
        ]
        data_records = [
            {"_id": "shared-recipe", "content": "Data content", "metadata": {"source": "data"}},
        ]
        result = combine_datasets(v2_records, data_records)
        assert len(result) == 1
        assert result[0]["content"] == "V2 content"
        assert result[0]["metadata"]["source"] == "v2"

    def test_combine_sorted_by_id(self):
        """Results should be sorted alphabetically by _id."""
        v2_records = [
            {"_id": "zebra", "content": "Z", "metadata": {}},
        ]
        data_records = [
            {"_id": "apple", "content": "A", "metadata": {}},
            {"_id": "banana", "content": "B", "metadata": {}},
        ]
        result = combine_datasets(v2_records, data_records)
        ids = [r["_id"] for r in result]
        assert ids == ["apple", "banana", "zebra"]


class TestCheckScoreThreshold:
    """Tests for the score threshold checking functionality."""

    def test_empty_results(self):
        passes, score, sorted_results = check_score_threshold([])
        assert passes is False
        assert score == 0.0
        assert sorted_results == []

    def test_single_result_above_threshold(self):
        results = [{"_id": "recipe-a", "_score": 0.85}]
        passes, score, sorted_results = check_score_threshold(results, threshold=0.7)
        assert passes is True
        assert score == 0.85
        assert len(sorted_results) == 1

    def test_single_result_below_threshold(self):
        results = [{"_id": "recipe-a", "_score": 0.5}]
        passes, score, sorted_results = check_score_threshold(results, threshold=0.7)
        assert passes is False
        assert score == 0.5

    def test_single_result_at_threshold(self):
        results = [{"_id": "recipe-a", "_score": 0.7}]
        passes, score, sorted_results = check_score_threshold(results, threshold=0.7)
        assert passes is True
        assert score == 0.7

    def test_multiple_results_sorted_correctly(self):
        """Should sort results by score descending and return highest."""
        results = [
            {"_id": "recipe-low", "_score": 0.3},
            {"_id": "recipe-high", "_score": 0.9},
            {"_id": "recipe-mid", "_score": 0.6},
        ]
        passes, score, sorted_results = check_score_threshold(results, threshold=0.7)
        assert passes is True
        assert score == 0.9
        assert sorted_results[0]["_id"] == "recipe-high"
        assert sorted_results[1]["_id"] == "recipe-mid"
        assert sorted_results[2]["_id"] == "recipe-low"

    def test_custom_threshold(self):
        results = [{"_id": "recipe-a", "_score": 0.55}]
        passes_low, _, _ = check_score_threshold(results, threshold=0.5)
        passes_high, _, _ = check_score_threshold(results, threshold=0.6)
        assert passes_low is True
        assert passes_high is False

    def test_missing_score_defaults_to_zero(self):
        results = [{"_id": "recipe-a"}]  # No _score key
        passes, score, _ = check_score_threshold(results, threshold=0.7)
        assert passes is False
        assert score == 0


class TestFormatRecipesForPrompt:
    """Tests for formatting RAG results for LLM prompts."""

    def test_empty_results(self):
        result = _format_recipes_for_prompt([])
        assert result == ""

    def test_single_result(self):
        results = [
            {
                "_id": "thai-chicken",
                "_score": 0.85,
                "fields": {
                    "title": "Thai Chicken",
                    "rating": 7.5,
                    "content": "# Thai Chicken\n\nDelicious recipe...",
                },
            }
        ]
        result = _format_recipes_for_prompt(results)
        assert "Recipe 1" in result
        assert "0.85" in result
        assert "Thai Chicken" in result
        assert "7.5/10" in result

    def test_respects_max_recipes(self):
        results = [
            {"_id": f"recipe-{i}", "_score": 0.9 - i * 0.1, "fields": {"content": f"Content {i}"}}
            for i in range(5)
        ]
        result = _format_recipes_for_prompt(results, max_recipes=2)
        assert "Recipe 1" in result
        assert "Recipe 2" in result
        assert "Recipe 3" not in result


class TestResponseFormatter:
    """Tests for the CLI response formatter."""

    def test_format_response_rag_source(self):
        result = format_response(
            "Here is your recipe...",
            RecipeSource.RAG_DATABASE,
            score=0.85,
        )
        assert "Recipe from your collection" in result
        assert "0.85" in result
        assert "Here is your recipe..." in result

    def test_format_response_generated_source(self):
        result = format_response(
            "Here is a generated recipe...",
            RecipeSource.LLM_GENERATED,
        )
        assert "Generated recipe" in result
        assert "no match found" in result
        assert "Here is a generated recipe..." in result

    def test_format_response_rag_without_score(self):
        result = format_response(
            "Recipe content",
            RecipeSource.RAG_DATABASE,
        )
        assert "Recipe from your collection" in result
        assert "relevance:" not in result

    def test_format_response_sparse_source(self):
        result = format_response(
            "Sparse recipe content",
            RecipeSource.RAG_SPARSE,
            score=0.42,
        )
        assert "sparse search" in result
        assert "0.42" in result

    def test_format_error(self):
        result = format_error("Something went wrong")
        assert "Error:" in result
        assert "Something went wrong" in result

    def test_format_welcome(self):
        result = format_welcome()
        assert "Recipe Chatbot" in result
        assert "quit" in result.lower()


class TestGenerateRecipeResponse:
    """Tests for LLM recipe response generation (mocked)."""

    def test_generate_recipe_response_calls_openai(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Here is your Thai Chicken recipe..."))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        rag_results = [
            {
                "_id": "thai-chicken",
                "_score": 0.85,
                "fields": {"title": "Thai Chicken", "content": "# Thai Chicken..."},
            }
        ]

        result = generate_recipe_response(
            "Give me a chicken recipe",
            rag_results,
            mock_client,
        )

        assert result == "Here is your Thai Chicken recipe..."
        mock_client.chat.completions.create.assert_called_once()

    def test_generate_recipe_response_empty_results(self):
        mock_client = MagicMock()

        result = generate_recipe_response("Any recipe", [], mock_client)

        assert "No recipes found" in result
        mock_client.chat.completions.create.assert_not_called()


class TestGenerateFallbackRecipe:
    """Tests for LLM fallback recipe generation (mocked)."""

    def test_generate_fallback_recipe_calls_openai(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="# Easy Chicken Stir Fry\n\n..."))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_fallback_recipe(
            "Easy chicken dinner",
            mock_client,
        )

        assert result == "# Easy Chicken Stir Fry\n\n..."
        mock_client.chat.completions.create.assert_called_once()

        # Verify the prompt mentions generating a recipe
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "Easy chicken dinner" in user_message["content"]


class TestProcessQuery:
    """Tests for dense/sparse query routing."""

    def test_dense_match_skips_sparse(self):
        from main import process_query

        dense_hits = [{"_id": "recipe-a", "_score": 0.9, "fields": {"content": "A"}}]
        sparse_hits = [{"_id": "recipe-b", "_score": 0.8, "fields": {"content": "B"}}]

        with patch("main.embed_text", return_value=[0.1]), \
             patch("main.search_dense_recipes", return_value=dense_hits) as dense_search, \
             patch("main.search_sparse_recipes", return_value=sparse_hits) as sparse_search, \
             patch("main.generate_recipe_response", return_value="RAG"), \
             patch("main.generate_fallback_recipe", return_value="LLM"):
            result = process_query(
                "test query",
                MagicMock(),
                "namespace",
                MagicMock(),
                "model",
                MagicMock(),
                threshold=0.1,
                sparse_threshold=0.0,
                min_dense_hits=1,
                dense_top_k=10,
                sparse_top_k=10,
            )

        dense_search.assert_called_once()
        sparse_search.assert_not_called()
        assert "Recipe from your collection" in result
        assert "sparse search" not in result

    def test_sparse_fallback_used_when_dense_insufficient(self):
        from main import process_query

        dense_hits = [{"_id": "recipe-a", "_score": 0.9, "fields": {"content": "A"}}]
        sparse_hits = [{"_id": "recipe-b", "_score": 0.8, "fields": {"content": "B"}}]

        with patch("main.embed_text", return_value=[0.1]), \
             patch("main.search_dense_recipes", return_value=dense_hits), \
             patch("main.search_sparse_recipes", return_value=sparse_hits), \
             patch("main.generate_recipe_response", return_value="RAG"), \
             patch("main.generate_fallback_recipe", return_value="LLM"):
            result = process_query(
                "test query",
                MagicMock(),
                "namespace",
                MagicMock(),
                "model",
                MagicMock(),
                threshold=0.1,
                sparse_threshold=0.0,
                min_dense_hits=2,
                dense_top_k=10,
                sparse_top_k=10,
            )

        assert "sparse search" in result

    def test_dense_fallback_used_when_sparse_missing(self):
        from main import process_query

        dense_hits = [{"_id": "recipe-a", "_score": 0.9, "fields": {"content": "A"}}]

        with patch("main.embed_text", return_value=[0.1]), \
             patch("main.search_dense_recipes", return_value=dense_hits), \
             patch("main.search_sparse_recipes", return_value=[]), \
             patch("main.generate_recipe_response", return_value="RAG"), \
             patch("main.generate_fallback_recipe", return_value="LLM"):
            result = process_query(
                "test query",
                MagicMock(),
                "namespace",
                MagicMock(),
                "model",
                MagicMock(),
                threshold=0.1,
                sparse_threshold=0.0,
                min_dense_hits=2,
                dense_top_k=10,
                sparse_top_k=10,
            )

        assert "Recipe from your collection" in result
        assert "sparse search" not in result

    def test_llm_fallback_when_no_hits(self):
        from main import process_query

        with patch("main.embed_text", return_value=[0.1]), \
             patch("main.search_dense_recipes", return_value=[]), \
             patch("main.search_sparse_recipes", return_value=[]), \
             patch("main.generate_recipe_response", return_value="RAG"), \
             patch("main.generate_fallback_recipe", return_value="LLM"):
            result = process_query(
                "test query",
                MagicMock(),
                "namespace",
                MagicMock(),
                "model",
                MagicMock(),
                threshold=0.1,
                sparse_threshold=0.0,
                min_dense_hits=2,
                dense_top_k=10,
                sparse_top_k=10,
            )

        assert "Generated recipe" in result


class TestEmbedRecordsFlattening:
    """Tests for metadata flattening in embed_records."""

    def test_flattens_nested_metadata(self):
        """Pinecone requires flat metadata, so nested metadata dict should be flattened."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3])
        ]
        mock_client.embeddings.create.return_value = mock_response

        records = [
            {
                "_id": "test-recipe",
                "content": "Recipe content here",
                "metadata": {
                    "title": "Test Recipe",
                    "cuisine": ["thai"],
                    "rating": 7.5,
                },
            }
        ]

        result = embed_records(records, "test-model", mock_client)

        assert len(result) == 1
        record_id, embedding, flat_metadata = result[0]

        assert record_id == "test-recipe"
        assert embedding == [0.1, 0.2, 0.3]

        # Verify metadata is flattened (no nested 'metadata' key)
        assert "metadata" not in flat_metadata
        assert flat_metadata["content"] == "Recipe content here"
        assert flat_metadata["title"] == "Test Recipe"
        assert flat_metadata["cuisine"] == ["thai"]
        assert flat_metadata["rating"] == 7.5

    def test_handles_record_without_nested_metadata(self):
        """Records without a nested metadata dict should still work."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        mock_client.embeddings.create.return_value = mock_response

        records = [
            {
                "_id": "simple-recipe",
                "content": "Simple content",
            }
        ]

        result = embed_records(records, "test-model", mock_client)

        _, _, flat_metadata = result[0]
        assert flat_metadata["content"] == "Simple content"
        assert "metadata" not in flat_metadata

    def test_filters_out_none_values_and_empty_arrays(self):
        """Pinecone doesn't accept None or empty arrays, so they should be filtered out."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        mock_client.embeddings.create.return_value = mock_response

        records = [
            {
                "_id": "test-recipe",
                "content": "Content",
                "metadata": {
                    "title": "Test",
                    "rating": None,  # Should be filtered
                    "diet": [],  # Should be filtered
                    "cuisine": ["thai"],  # Should be kept
                    "difficulty": "easy",  # Should be kept
                },
            }
        ]

        result = embed_records(records, "test-model", mock_client)

        _, _, flat_metadata = result[0]
        assert flat_metadata["title"] == "Test"
        assert flat_metadata["cuisine"] == ["thai"]
        assert flat_metadata["difficulty"] == "easy"
        assert "rating" not in flat_metadata  # None filtered out
        assert "diet" not in flat_metadata  # Empty array filtered out
