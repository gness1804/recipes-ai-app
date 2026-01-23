"""
Tests for the vector database preparation scripts.

Tests cover:
- Recipe markdown parsing
- LLM classification (with mocked API calls)
- Record creation
- Cache functionality
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_for_vector_db import (
    create_record,
    parse_recipe_markdown,
)


class TestParseRecipeMarkdown:
    """Tests for parsing processed recipe markdown files."""

    def test_parse_basic_recipe(self, tmp_path):
        recipe_content = """# Thai Chicken

Rating: 7/10

## Ingredients

- 2.5 cups instant rice
- 4 chicken breast halves
- 2 teaspoons vegetable oil

## Instructions

1. Bring water to a boil.
2. Cook the chicken.
3. Serve over rice.
"""
        recipe_file = tmp_path / "thai-chicken.md"
        recipe_file.write_text(recipe_content)

        result = parse_recipe_markdown(recipe_file)

        assert result["title"] == "Thai Chicken"
        assert result["rating"] == 7.0
        assert "2.5 cups instant rice" in result["ingredients"]
        assert "Bring water to a boil" in result["instructions"]
        assert result["content"] == recipe_content

    def test_parse_recipe_with_decimal_rating(self, tmp_path):
        recipe_content = """# Test Recipe

Rating: 7.5/10

## Ingredients

- 1 cup flour

## Instructions

1. Mix ingredients.
"""
        recipe_file = tmp_path / "test-recipe.md"
        recipe_file.write_text(recipe_content)

        result = parse_recipe_markdown(recipe_file)

        assert result["rating"] == 7.5

    def test_parse_recipe_with_missing_rating(self, tmp_path):
        recipe_content = """# Test Recipe

Rating: [MISSING]

## Ingredients

- 1 cup flour

## Instructions

1. Mix ingredients.
"""
        recipe_file = tmp_path / "test-recipe.md"
        recipe_file.write_text(recipe_content)

        result = parse_recipe_markdown(recipe_file)

        assert result["rating"] is None

    def test_parse_recipe_with_n_rating(self, tmp_path):
        recipe_content = """# Test Recipe

Rating: N

## Ingredients

- 1 cup flour

## Instructions

1. Mix ingredients.
"""
        recipe_file = tmp_path / "test-recipe.md"
        recipe_file.write_text(recipe_content)

        result = parse_recipe_markdown(recipe_file)

        assert result["rating"] is None

    def test_parse_recipe_with_subsections(self, tmp_path):
        recipe_content = """# Complex Recipe

Rating: 8/10

## Ingredients

### Main Ingredients
- 1 lb chicken
- 2 cups rice

### For the Sauce
- 0.5 cup soy sauce
- 2 tablespoons honey

## Instructions

### Main Dish
1. Cook the chicken.
2. Prepare the rice.

### For the Sauce
1. Mix soy sauce and honey.
"""
        recipe_file = tmp_path / "complex-recipe.md"
        recipe_file.write_text(recipe_content)

        result = parse_recipe_markdown(recipe_file)

        assert result["title"] == "Complex Recipe"
        assert "### Main Ingredients" in result["ingredients"]
        assert "### For the Sauce" in result["ingredients"]
        assert "### Main Dish" in result["instructions"]

    def test_parse_recipe_no_rating_line(self, tmp_path):
        recipe_content = """# No Rating Recipe

## Ingredients

- 1 cup flour

## Instructions

1. Mix ingredients.
"""
        recipe_file = tmp_path / "no-rating.md"
        recipe_file.write_text(recipe_content)

        result = parse_recipe_markdown(recipe_file)

        assert result["rating"] is None


class TestCreateRecord:
    """Tests for creating vector DB records."""

    def test_create_record_basic(self):
        parsed = {
            "title": "Thai Chicken",
            "ingredients": "- 2 cups rice\n- 1 lb chicken",
            "instructions": "1. Cook rice\n2. Cook chicken",
            "rating": 7.5,
            "content": "Full recipe content here",
        }
        classification = {
            "diet": [],
            "protein": ["chicken"],
            "cuisine": ["thai"],
            "meal_type": ["main-dish"],
            "difficulty": "easy",
            "prepTimeMinutes": 30,
        }

        record = create_record("thai-chicken", parsed, classification)

        assert record["_id"] == "thai-chicken"
        assert record["content"] == "Full recipe content here"
        assert record["metadata"]["title"] == "Thai Chicken"
        assert record["metadata"]["rating"] == 7.5
        assert record["metadata"]["protein"] == ["chicken"]
        assert record["metadata"]["cuisine"] == ["thai"]
        assert record["metadata"]["difficulty"] == "easy"
        assert record["metadata"]["prepTimeMinutes"] == 30

    def test_create_record_with_null_values(self):
        parsed = {
            "title": "Mystery Recipe",
            "ingredients": "- Unknown",
            "instructions": "1. Figure it out",
            "rating": None,
            "content": "Content",
        }
        classification = {
            "diet": [],
            "protein": ["none"],
            "cuisine": ["other"],
            "meal_type": ["main-dish"],
            "difficulty": "medium",
            "prepTimeMinutes": None,
        }

        record = create_record("mystery-recipe", parsed, classification)

        assert record["metadata"]["rating"] is None
        assert record["metadata"]["prepTimeMinutes"] is None


class TestLLMClassifier:
    """Tests for the LLM classifier module with mocked API calls."""

    def test_classify_recipe_returns_expected_structure(self, tmp_path, monkeypatch):
        from llm_classifier import classify_recipe, CACHE_FILE

        # Mock the cache file location
        cache_file = tmp_path / "test_cache.json"
        monkeypatch.setattr("llm_classifier.CACHE_FILE", cache_file)
        monkeypatch.setattr("llm_classifier.CACHE_DIR", tmp_path)

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "diet": ["gluten-free"],
                            "protein": ["chicken"],
                            "cuisine": ["thai"],
                            "meal_type": ["main-dish"],
                            "difficulty": "easy",
                            "prepTimeMinutes": 30,
                        }
                    )
                )
            )
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        monkeypatch.setattr("llm_classifier.client", mock_client)

        result = classify_recipe(
            "test-recipe",
            "# Thai Chicken\n\n## Ingredients\n- chicken",
            use_cache=False,
        )

        assert result["diet"] == ["gluten-free"]
        assert result["protein"] == ["chicken"]
        assert result["cuisine"] == ["thai"]
        assert result["difficulty"] == "easy"
        assert result["prepTimeMinutes"] == 30

    def test_classify_recipe_uses_cache(self, tmp_path, monkeypatch):
        from llm_classifier import classify_recipe, load_cache, save_cache

        # Set up cache location
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "classifications.json"
        monkeypatch.setattr("llm_classifier.CACHE_FILE", cache_file)
        monkeypatch.setattr("llm_classifier.CACHE_DIR", cache_dir)

        # Pre-populate cache
        cached_data = {
            "cached-recipe": {
                "diet": ["vegan"],
                "protein": ["tofu"],
                "cuisine": ["chinese"],
                "meal_type": ["main-dish"],
                "difficulty": "medium",
                "prepTimeMinutes": 45,
            }
        }
        cache_file.write_text(json.dumps(cached_data))

        # Mock the client to track if it's called
        mock_client = MagicMock()
        monkeypatch.setattr("llm_classifier.client", mock_client)

        result = classify_recipe(
            "cached-recipe",
            "# Tofu Stir Fry\n\n## Ingredients\n- tofu",
            use_cache=True,
        )

        # Should return cached result without calling API
        mock_client.chat.completions.create.assert_not_called()
        assert result["protein"] == ["tofu"]
        assert result["cuisine"] == ["chinese"]

    def test_classify_recipe_handles_markdown_fenced_response(
        self, tmp_path, monkeypatch
    ):
        from llm_classifier import classify_recipe

        # Set up cache location
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "classifications.json"
        monkeypatch.setattr("llm_classifier.CACHE_FILE", cache_file)
        monkeypatch.setattr("llm_classifier.CACHE_DIR", cache_dir)

        # Mock response with markdown code fences
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="""```json
{
    "diet": [],
    "protein": ["seafood"],
    "cuisine": ["cajun"],
    "meal_type": ["main-dish"],
    "difficulty": "medium",
    "prepTimeMinutes": 60
}
```"""
                )
            )
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        monkeypatch.setattr("llm_classifier.client", mock_client)

        result = classify_recipe(
            "gumbo",
            "# Gumbo\n\n## Ingredients\n- shrimp",
            use_cache=False,
        )

        assert result["protein"] == ["seafood"]
        assert result["cuisine"] == ["cajun"]


class TestDiscoveredEnums:
    """Tests for dynamic enum expansion."""

    def test_discovered_enums_tracked(self, tmp_path, monkeypatch):
        from llm_classifier import (
            classify_recipe,
            get_discovered_enums,
            discovered_enums,
            BASE_ENUMS,
        )

        # Reset discovered enums
        for key in discovered_enums:
            discovered_enums[key].clear()

        # Set up cache location
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "classifications.json"
        monkeypatch.setattr("llm_classifier.CACHE_FILE", cache_file)
        monkeypatch.setattr("llm_classifier.CACHE_DIR", cache_dir)

        # Mock response with a new cuisine value
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "diet": [],
                            "protein": ["duck"],  # New protein not in base
                            "cuisine": ["vietnamese"],  # New cuisine not in base
                            "meal_type": ["main-dish"],
                            "difficulty": "hard",
                            "prepTimeMinutes": 120,
                        }
                    )
                )
            )
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        monkeypatch.setattr("llm_classifier.client", mock_client)

        classify_recipe(
            "pho",
            "# Vietnamese Pho\n\n## Ingredients\n- duck",
            use_cache=False,
        )

        discovered = get_discovered_enums()

        assert "duck" in discovered.get("protein", [])
        assert "vietnamese" in discovered.get("cuisine", [])


class TestBaseEnums:
    """Tests for base enum values."""

    def test_base_enums_defined(self):
        from llm_classifier import BASE_ENUMS

        assert "vegetarian" in BASE_ENUMS["diet"]
        assert "chicken" in BASE_ENUMS["protein"]
        assert "thai" in BASE_ENUMS["cuisine"]
        assert "main-dish" in BASE_ENUMS["meal_type"]
        assert "easy" in BASE_ENUMS["difficulty"]

    def test_get_all_known_values(self, monkeypatch):
        from llm_classifier import get_all_known_values, discovered_enums

        # Add a discovered value
        discovered_enums["cuisine"].add("vietnamese")

        values = get_all_known_values("cuisine")

        assert "thai" in values  # Base value
        assert "vietnamese" in values  # Discovered value
