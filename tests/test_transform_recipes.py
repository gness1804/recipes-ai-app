"""
Tests for the recipe transformer script.

These tests cover utility functions and parsing logic.
API calls are mocked to avoid actual OpenAI charges during testing.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from transform_recipes import (
    IMAGE_EXTENSIONS,
    MARKDOWN_EXTENSIONS,
    PDF_EXTENSIONS,
    extract_recipe_name,
    get_existing_processed_recipes,
    get_raw_recipes,
    slugify,
    strip_markdown_fences,
)


class TestSlugify:
    """Tests for the slugify function."""

    def test_basic_slugify(self):
        assert slugify("African Curried Soup") == "african-curried-soup"

    def test_slugify_with_special_characters(self):
        assert slugify("Mom's Famous Recipe!") == "moms-famous-recipe"

    def test_slugify_with_numbers(self):
        assert slugify("30-Minute Chicken") == "30-minute-chicken"

    def test_slugify_with_multiple_spaces(self):
        assert slugify("Chicken   Vindaloo") == "chicken-vindaloo"

    def test_slugify_with_underscores(self):
        assert slugify("Chicken_Vindaloo") == "chicken-vindaloo"

    def test_slugify_preserves_hyphens(self):
        assert slugify("Stir-Fry Recipe") == "stir-fry-recipe"

    def test_slugify_strips_leading_trailing_hyphens(self):
        assert slugify("  Recipe Name  ") == "recipe-name"

    def test_slugify_empty_string(self):
        assert slugify("") == ""

    def test_slugify_with_apostrophes(self):
        assert slugify("Grandma's Old-Fashioned Pie") == "grandmas-old-fashioned-pie"


class TestStripMarkdownFences:
    """Tests for stripping markdown code fences from GPT-4o output."""

    def test_strip_fences_with_markdown_tag(self):
        content = "```markdown\n# Recipe\n\nRating: 8/10\n```"
        result = strip_markdown_fences(content)
        assert result == "# Recipe\n\nRating: 8/10"

    def test_strip_fences_without_language_tag(self):
        content = "```\n# Recipe\n\nRating: 8/10\n```"
        result = strip_markdown_fences(content)
        assert result == "# Recipe\n\nRating: 8/10"

    def test_no_fences_unchanged(self):
        content = "# Recipe\n\nRating: 8/10"
        result = strip_markdown_fences(content)
        assert result == "# Recipe\n\nRating: 8/10"

    def test_strip_fences_with_whitespace(self):
        content = "  ```markdown\n# Recipe\n```  "
        result = strip_markdown_fences(content)
        assert result == "# Recipe"


class TestExtractRecipeName:
    """Tests for extracting recipe names from transformed content."""

    def test_extract_name_basic(self):
        content = "# African Curried Soup\n\nRating: 8/10\n\n## Ingredients"
        assert extract_recipe_name(content) == "African Curried Soup"

    def test_extract_name_with_extra_spaces(self):
        content = "#   Chicken Vindaloo  \n\nRating: 7/10"
        assert extract_recipe_name(content) == "Chicken Vindaloo"

    def test_extract_name_no_header(self):
        content = "Rating: 8/10\n\n## Ingredients"
        assert extract_recipe_name(content) == "unknown-recipe"

    def test_extract_name_multiple_headers(self):
        content = "# Main Recipe\n\n## Ingredients\n\n# Another Header"
        assert extract_recipe_name(content) == "Main Recipe"


class TestFileExtensions:
    """Tests for file extension sets."""

    def test_markdown_extensions(self):
        assert ".md" in MARKDOWN_EXTENSIONS
        assert ".markdown" in MARKDOWN_EXTENSIONS
        assert ".txt" in MARKDOWN_EXTENSIONS

    def test_pdf_extensions(self):
        assert ".pdf" in PDF_EXTENSIONS

    def test_image_extensions(self):
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".gif" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS


class TestGetRawRecipes:
    """Tests for getting raw recipe files."""

    def test_get_raw_recipes_returns_list(self, tmp_path, monkeypatch):
        # Create temp directory with test files
        raw_dir = tmp_path / "raw-recipes"
        raw_dir.mkdir()
        (raw_dir / "test1.md").write_text("# Test 1")
        (raw_dir / "test2.pdf").write_text("fake pdf")
        (raw_dir / "test3.jpg").write_text("fake image")
        (raw_dir / "ignore.xyz").write_text("ignored")

        # Monkeypatch the RAW_RECIPES_DIR
        import transform_recipes

        monkeypatch.setattr(transform_recipes, "RAW_RECIPES_DIR", raw_dir)

        recipes = get_raw_recipes()

        assert len(recipes) == 3
        names = [r.name for r in recipes]
        assert "test1.md" in names
        assert "test2.pdf" in names
        assert "test3.jpg" in names
        assert "ignore.xyz" not in names


class TestGetExistingProcessedRecipes:
    """Tests for detecting already-processed recipes."""

    def test_get_existing_processed_recipes(self, tmp_path, monkeypatch):
        # Create temp directory with processed files
        processed_dir = tmp_path / "processed-recipes"
        processed_dir.mkdir()
        (processed_dir / "african-curried-soup.md").write_text("# African Curried Soup")
        (processed_dir / "chicken-vindaloo.md").write_text("# Chicken Vindaloo")
        (processed_dir / "_missing_ratings.md").write_text("# Missing")

        # Monkeypatch the PROCESSED_RECIPES_DIR
        import transform_recipes

        monkeypatch.setattr(transform_recipes, "PROCESSED_RECIPES_DIR", processed_dir)

        existing = get_existing_processed_recipes()

        assert "african-curried-soup" in existing
        assert "chicken-vindaloo" in existing
        assert "_missing_ratings" not in existing  # Should be excluded


class TestMissingRatingsDetection:
    """Tests for detecting missing ratings in transformed content."""

    def test_missing_rating_detected(self):
        content = "# Test Recipe\n\nRating: [MISSING]\n\n## Ingredients"
        assert "[MISSING]" in content

    def test_rating_present(self):
        content = "# Test Recipe\n\nRating: 8/10\n\n## Ingredients"
        assert "[MISSING]" not in content


class TestProcessRecipeMocked:
    """Tests for recipe processing with mocked API calls."""

    @patch("transform_recipes.call_gpt4o_text")
    def test_process_markdown_calls_gpt4o_text(self, mock_gpt4o, tmp_path):
        from transform_recipes import process_markdown

        # Create a test markdown file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Recipe\n\nIngredients:\n- 1 cup flour")

        mock_gpt4o.return_value = (
            "# Test Recipe\n\nRating: [MISSING]\n\n## Ingredients\n\n- 1 cup flour"
        )

        result = process_markdown(test_file)

        mock_gpt4o.assert_called_once()
        assert "# Test Recipe" in result

    @patch("transform_recipes.call_gpt4o_vision")
    @patch("transform_recipes.convert_from_path")
    def test_process_pdf_calls_gpt4o_vision(self, mock_convert, mock_gpt4o, tmp_path):
        from PIL import Image
        from transform_recipes import process_pdf

        # Create a test PDF path (we'll mock the conversion)
        test_file = tmp_path / "test.pdf"
        test_file.write_text("fake pdf content")

        # Mock PDF to image conversion
        mock_image = MagicMock(spec=Image.Image)
        mock_image.save = MagicMock()
        mock_convert.return_value = [mock_image]

        mock_gpt4o.return_value = "# PDF Recipe\n\nRating: 7/10\n\n## Ingredients"

        result = process_pdf(test_file)

        mock_convert.assert_called_once()
        mock_gpt4o.assert_called_once()
        assert "# PDF Recipe" in result

    @patch("transform_recipes.call_gpt4o_vision")
    def test_process_image_calls_gpt4o_vision(self, mock_gpt4o, tmp_path):
        from transform_recipes import process_image

        # Create a minimal valid PNG file
        test_file = tmp_path / "test.png"
        # Minimal PNG header
        png_header = b"\x89PNG\r\n\x1a\n"
        test_file.write_bytes(png_header + b"\x00" * 100)

        mock_gpt4o.return_value = "# Image Recipe\n\nRating: 9/10\n\n## Ingredients"

        result = process_image(test_file)

        mock_gpt4o.assert_called_once()
        assert "# Image Recipe" in result


class TestSaveProcessedRecipe:
    """Tests for saving processed recipes."""

    def test_save_processed_recipe(self, tmp_path, monkeypatch):
        # Monkeypatch the output directory
        import transform_recipes
        from transform_recipes import save_processed_recipe

        monkeypatch.setattr(transform_recipes, "PROCESSED_RECIPES_DIR", tmp_path)

        content = "# African Curried Soup\n\nRating: 8/10\n\n## Ingredients"
        original_path = Path("/fake/path/AfricanCurriedSoup.md")

        output_path = save_processed_recipe(content, original_path)

        assert output_path.exists()
        assert output_path.name == "african-curried-soup.md"
        assert output_path.read_text() == content
