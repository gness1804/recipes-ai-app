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
    IMAGE_EXTENSIONS,
    TEXT_EXTENSIONS,
    PDF_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
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


class TestSupportedExtensions:
    """Tests for supported file extensions."""

    def test_pdf_in_supported_extensions(self):
        """PDF should be in SUPPORTED_EXTENSIONS."""
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_pdf_extensions_defined(self):
        """PDF_EXTENSIONS should be defined with .pdf."""
        assert ".pdf" in PDF_EXTENSIONS

    def test_all_extensions_in_supported(self):
        """All extension sets should be included in SUPPORTED_EXTENSIONS."""
        assert IMAGE_EXTENSIONS.issubset(SUPPORTED_EXTENSIONS)
        assert TEXT_EXTENSIONS.issubset(SUPPORTED_EXTENSIONS)
        assert PDF_EXTENSIONS.issubset(SUPPORTED_EXTENSIONS)


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

    @patch("v2.process_recipes.b")
    @patch("v2.process_recipes.convert_from_path")
    @patch("v2.process_recipes.baml_py")
    def test_extract_recipe_from_pdf_single_page(self, mock_baml_py, mock_convert, mock_b):
        """Test that single-page PDF extraction uses ExtractRecipeFromImage."""
        from v2.process_recipes import extract_recipe_from_pdf
        from PIL import Image
        import tempfile

        mock_recipe = MagicMock()
        mock_b.ExtractRecipeFromImage.return_value = mock_recipe

        # Create a mock single-page PDF (mock the conversion)
        mock_image = Image.new("RGB", (100, 100), color="white")
        mock_convert.return_value = [mock_image]

        # Create a temp PDF file (just need the path, content is mocked)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = extract_recipe_from_pdf(temp_path)
            mock_convert.assert_called_once()
            mock_b.ExtractRecipeFromImage.assert_called_once()
            assert result == mock_recipe
        finally:
            temp_path.unlink()

    @patch("v2.process_recipes.b")
    @patch("v2.process_recipes.convert_from_path")
    @patch("v2.process_recipes.baml_py")
    def test_extract_recipe_from_pdf_multi_page(self, mock_baml_py, mock_convert, mock_b):
        """Test that multi-page PDF extraction uses ExtractRecipeFromImages."""
        from v2.process_recipes import extract_recipe_from_pdf
        from PIL import Image
        import tempfile

        mock_recipe = MagicMock()
        mock_b.ExtractRecipeFromImages.return_value = mock_recipe

        # Create mock multi-page PDF (mock the conversion)
        mock_images = [
            Image.new("RGB", (100, 100), color="white"),
            Image.new("RGB", (100, 100), color="gray"),
        ]
        mock_convert.return_value = mock_images

        # Create a temp PDF file (just need the path, content is mocked)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = extract_recipe_from_pdf(temp_path)
            mock_convert.assert_called_once()
            mock_b.ExtractRecipeFromImages.assert_called_once()
            assert result == mock_recipe
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
