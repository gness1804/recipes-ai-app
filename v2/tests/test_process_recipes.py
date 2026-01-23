"""
Tests for V2 recipe processing.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v2.process_recipes import (
    slugify,
    transform_to_vector_db_format,
    format_recipe_dict,
)


class TestSlugify:
    """Tests for the slugify function."""

    def test_basic_slugify(self):
        assert slugify("Caribbean Chicken.jpg") == "caribbean-chicken"

    def test_slugify_with_underscores(self):
        assert slugify("Ultimate_Chocolate_Chip_Cookie.jpg") == "ultimate-chocolate-chip-cookie"

    def test_slugify_mixed(self):
        assert slugify("My Recipe_Name Here.md") == "my-recipe-name-here"

    def test_slugify_multiple_spaces(self):
        assert slugify("Recipe   With   Spaces.txt") == "recipe-with-spaces"

    def test_slugify_already_lowercase(self):
        assert slugify("simple-recipe.md") == "simple-recipe"


class TestTransformToVectorDbFormat:
    """Tests for transforming BAML output to vector DB format."""

    @pytest.fixture
    def sample_recipe(self):
        """Create a mock Recipe object."""
        recipe = MagicMock()
        recipe.id = "test-recipe"
        recipe.title = "Test Recipe"
        recipe.ingredients = "1. 1 cup flour\n2. 2 cups sugar"
        recipe.instructions = "1. Mix ingredients.\n2. Bake at 350F."
        recipe.rating = 7.5
        recipe.diet = ["vegetarian"]
        recipe.protein = ["none"]
        recipe.cuisine = ["american"]
        recipe.meal_type = ["dessert"]
        recipe.difficulty = "easy"
        recipe.prepTimeMinutes = 30
        return recipe

    def test_transform_basic(self, sample_recipe):
        result = transform_to_vector_db_format(sample_recipe)

        assert result["_id"] == "test-recipe"
        assert "# Test Recipe" in result["content"]
        assert "Rating: 7.5/10" in result["content"]
        assert result["metadata"]["title"] == "Test Recipe"
        assert result["metadata"]["rating"] == 7.5
        assert result["metadata"]["diet"] == ["vegetarian"]

    def test_transform_null_rating(self, sample_recipe):
        sample_recipe.rating = None
        result = transform_to_vector_db_format(sample_recipe)

        assert "Rating: N" in result["content"]
        assert result["metadata"]["rating"] is None

    def test_transform_ingredients_format(self, sample_recipe):
        result = transform_to_vector_db_format(sample_recipe)

        # Should convert numbered list to bullet list in metadata
        assert "- 1 cup flour" in result["metadata"]["ingredients"]
        assert "- 2 cups sugar" in result["metadata"]["ingredients"]

    def test_transform_preserves_arrays(self, sample_recipe):
        sample_recipe.diet = ["vegetarian", "gluten-free"]
        sample_recipe.cuisine = ["indian", "thai"]

        result = transform_to_vector_db_format(sample_recipe)

        assert result["metadata"]["diet"] == ["vegetarian", "gluten-free"]
        assert result["metadata"]["cuisine"] == ["indian", "thai"]


class TestFormatRecipeDict:
    """Tests for formatting recipe dicts as Python code."""

    def test_format_basic(self):
        recipe = {
            "_id": "test",
            "content": "Test content",
            "metadata": {
                "title": "Test",
                "ingredients": "- flour",
                "instructions": "1. Mix",
                "rating": 5.0,
                "diet": [],
                "protein": ["none"],
                "cuisine": ["american"],
                "meal_type": ["main-dish"],
                "difficulty": "easy",
                "prepTimeMinutes": 30,
            }
        }

        result = format_recipe_dict(recipe, indent=4)

        assert '"_id": \'test\'' in result
        assert '"title": \'Test\'' in result
        assert '"rating": 5.0' in result

    def test_format_handles_special_characters(self):
        recipe = {
            "_id": "test",
            "content": "Line 1\nLine 2\tTabbed",
            "metadata": {
                "title": "Test's Recipe",
                "ingredients": "1 cup \"fancy\" flour",
                "instructions": "Mix it",
                "rating": None,
                "diet": [],
                "protein": [],
                "cuisine": [],
                "meal_type": [],
                "difficulty": "easy",
                "prepTimeMinutes": 15,
            }
        }

        result = format_recipe_dict(recipe, indent=0)

        # Should escape properly for Python
        assert "\\n" in result or repr("Line 1\nLine 2") in result
        assert '"rating": None' in result


class TestIntegration:
    """Integration tests (require mocking BAML calls)."""

    @patch("v2.process_recipes.b")
    def test_extract_recipe_from_text(self, mock_b):
        """Test that text extraction calls the right BAML function."""
        from v2.process_recipes import extract_recipe_from_text

        mock_recipe = MagicMock()
        mock_b.ExtractRecipe.return_value = mock_recipe

        # Create a temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test recipe content")
            temp_path = Path(f.name)

        try:
            result = extract_recipe_from_text(temp_path)
            mock_b.ExtractRecipe.assert_called_once()
            assert result == mock_recipe
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
