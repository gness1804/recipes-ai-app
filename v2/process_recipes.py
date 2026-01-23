#!/usr/bin/env python3
"""
V2 Recipe Processing Script

Processes raw recipes using BAML to extract structured data and prepare
them for RAG vector database ingestion.

Usage:
    python v2/process_recipes.py              # Process new recipes only
    python v2/process_recipes.py --force      # Reprocess all recipes
    python v2/process_recipes.py --dry-run    # Show what would be processed
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import baml_py

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from baml_client.sync_client import b
from baml_client.types import Recipe

# Constants
V2_DIR = Path(__file__).parent
RAW_RECIPES_DIR = V2_DIR / "raw-recipes"
OUTPUT_FILE = V2_DIR / "recipes_for_vector_db.py"
MANIFEST_FILE = V2_DIR / "_processed_manifest.json"
UNRATED_FILE = V2_DIR / "_unrated.json"

# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
TEXT_EXTENSIONS = {".txt", ".md"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | TEXT_EXTENSIONS


def slugify(filename: str) -> str:
    """Convert a filename to a slug for tracking purposes."""
    # Remove extension and convert to lowercase slug
    name = Path(filename).stem
    # Replace spaces and underscores with hyphens, lowercase
    slug = name.lower().replace(" ", "-").replace("_", "-")
    # Remove any double hyphens
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug


def load_manifest() -> dict:
    """Load the processed manifest file."""
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    """Save the processed manifest file."""
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def load_existing_recipes() -> list:
    """Load existing recipes from the output file if it exists."""
    if not OUTPUT_FILE.exists():
        return []

    # Import the existing records
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("recipes", OUTPUT_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return list(module.RECIPE_RECORDS)
    except Exception as e:
        print(f"Warning: Could not load existing recipes: {e}")
        return []


def load_unrated() -> list:
    """Load the list of unrated recipes."""
    if UNRATED_FILE.exists():
        with open(UNRATED_FILE, "r") as f:
            return json.load(f)
    return []


def save_unrated(unrated: list) -> None:
    """Save the list of unrated recipes."""
    with open(UNRATED_FILE, "w") as f:
        json.dump(unrated, f, indent=2)


def get_raw_recipe_files() -> list[Path]:
    """Get all supported recipe files from the raw-recipes directory."""
    if not RAW_RECIPES_DIR.exists():
        return []

    files = []
    for file in RAW_RECIPES_DIR.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(file)

    return sorted(files)


def extract_recipe_from_image(image_path: Path) -> Recipe:
    """Extract recipe data from an image using BAML."""
    import base64

    # Read image file and convert to base64
    with open(image_path, "rb") as f:
        image_data = f.read()

    base64_data = base64.b64encode(image_data).decode("utf-8")

    # Determine media type from extension
    ext = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(ext, "image/jpeg")

    image = baml_py.Image.from_base64(media_type, base64_data)
    return b.ExtractRecipeFromImage(image)


def extract_recipe_from_text(text_path: Path) -> Recipe:
    """Extract recipe data from a text file using BAML."""
    with open(text_path, "r") as f:
        content = f.read()
    return b.ExtractRecipe(content)


def transform_to_vector_db_format(recipe: Recipe) -> dict:
    """Transform BAML Recipe output to basicSchema format for vector DB."""
    # Build the content string (full recipe text with rating)
    rating_str = f"{recipe.rating}/10" if recipe.rating is not None else "N"
    content = f"# {recipe.title}\n\nRating: {rating_str}\n\n## Ingredients\n\n{recipe.ingredients}\n\n## Instructions\n\n{recipe.instructions}"

    # Convert ingredients from numbered list to bullet list for metadata
    ingredients_lines = recipe.ingredients.strip().split("\n")
    ingredients_bullets = []
    for line in ingredients_lines:
        # Remove leading number and period/dot
        line = line.strip()
        if line and line[0].isdigit():
            # Find where the number ends
            i = 0
            while i < len(line) and (line[i].isdigit() or line[i] in ".)"):
                i += 1
            line = line[i:].strip()
        if line:
            ingredients_bullets.append(f"- {line}")
    ingredients_formatted = "\n".join(ingredients_bullets)

    return {
        "_id": recipe.id,
        "content": content,
        "metadata": {
            "title": recipe.title,
            "ingredients": ingredients_formatted,
            "instructions": recipe.instructions,
            "rating": recipe.rating,
            "diet": recipe.diet,
            "protein": recipe.protein,
            "cuisine": recipe.cuisine,
            "meal_type": recipe.meal_type,
            "difficulty": recipe.difficulty,
            "prepTimeMinutes": recipe.prepTimeMinutes,
        },
    }


def write_output_file(recipes: list) -> None:
    """Write recipes to the output Python file."""
    # Sort recipes by _id for consistency
    recipes = sorted(recipes, key=lambda r: r["_id"])

    output = '''"""
Recipe records prepared for Pinecone vector database ingestion.

Generated by: v2/process_recipes.py
Schema: data/schemas/basicSchema.json

Usage:
    from v2.recipes_for_vector_db import RECIPE_RECORDS
"""

RECIPE_RECORDS = [
'''

    for i, recipe in enumerate(recipes):
        # Format the recipe dict with proper indentation
        recipe_str = format_recipe_dict(recipe, indent=4)
        output += recipe_str
        if i < len(recipes) - 1:
            output += ",\n"
        else:
            output += "\n"

    output += "]\n"

    with open(OUTPUT_FILE, "w") as f:
        f.write(output)


def format_recipe_dict(recipe: dict, indent: int = 0) -> str:
    """Format a recipe dict as a Python literal with proper indentation."""
    ind = " " * indent
    ind2 = " " * (indent + 4)
    ind3 = " " * (indent + 8)

    lines = [f"{ind}{{"]

    # _id
    lines.append(f'{ind2}"_id": {repr(recipe["_id"])},')

    # content (multiline string)
    content = recipe["content"]
    lines.append(f'{ind2}"content": {repr(content)},')

    # metadata
    lines.append(f'{ind2}"metadata": {{')
    meta = recipe["metadata"]

    lines.append(f'{ind3}"title": {repr(meta["title"])},')
    lines.append(f'{ind3}"ingredients": {repr(meta["ingredients"])},')
    lines.append(f'{ind3}"instructions": {repr(meta["instructions"])},')
    lines.append(f'{ind3}"rating": {repr(meta["rating"])},')
    lines.append(f'{ind3}"diet": {repr(meta["diet"])},')
    lines.append(f'{ind3}"protein": {repr(meta["protein"])},')
    lines.append(f'{ind3}"cuisine": {repr(meta["cuisine"])},')
    lines.append(f'{ind3}"meal_type": {repr(meta["meal_type"])},')
    lines.append(f'{ind3}"difficulty": {repr(meta["difficulty"])},')
    lines.append(f'{ind3}"prepTimeMinutes": {repr(meta["prepTimeMinutes"])},')

    lines.append(f"{ind2}}},")
    lines.append(f"{ind}}}")

    return "\n".join(lines)


def process_recipes(force: bool = False, dry_run: bool = False) -> None:
    """Main processing function."""
    print("V2 Recipe Processing")
    print("=" * 40)

    # Load existing data
    manifest = load_manifest()
    existing_recipes = load_existing_recipes()
    existing_ids = {r["_id"] for r in existing_recipes}
    unrated = load_unrated()

    # Get files to process
    raw_files = get_raw_recipe_files()

    if not raw_files:
        print("No raw recipe files found in v2/raw-recipes/")
        return

    print(f"Found {len(raw_files)} raw recipe file(s)")

    # Determine which files need processing
    files_to_process = []
    for file in raw_files:
        file_slug = slugify(file.name)
        if force or file.name not in manifest:
            files_to_process.append(file)
        else:
            print(f"  Skipping (already processed): {file.name}")

    if not files_to_process:
        print("\nNo new recipes to process.")
        return

    print(f"\nWill process {len(files_to_process)} recipe(s):")
    for file in files_to_process:
        print(f"  - {file.name}")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    # Process each file
    new_recipes = []
    new_unrated = []

    for file in files_to_process:
        print(f"\nProcessing: {file.name}")

        try:
            # Extract recipe using appropriate method
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                recipe = extract_recipe_from_image(file)
            else:
                recipe = extract_recipe_from_text(file)

            print(f"  Title: {recipe.title}")
            print(f"  Rating: {recipe.rating}")
            print(f"  Cuisine: {recipe.cuisine}")

            # Transform to vector DB format
            record = transform_to_vector_db_format(recipe)

            # Check if this ID already exists (from a different source file)
            if record["_id"] in existing_ids:
                if force:
                    # Remove the old version
                    existing_recipes = [r for r in existing_recipes if r["_id"] != record["_id"]]
                    existing_ids.discard(record["_id"])
                else:
                    print(f"  Warning: Recipe ID '{record['_id']}' already exists, skipping")
                    continue

            new_recipes.append(record)
            existing_ids.add(record["_id"])

            # Track unrated recipes
            if recipe.rating is None:
                new_unrated.append({
                    "id": record["_id"],
                    "title": recipe.title,
                    "source_file": file.name,
                })

            # Update manifest
            manifest[file.name] = record["_id"]

        except Exception as e:
            print(f"  Error processing {file.name}: {e}")
            continue

    # Combine existing and new recipes
    all_recipes = existing_recipes + new_recipes

    # Write output
    print(f"\nWriting {len(all_recipes)} recipe(s) to {OUTPUT_FILE.name}")
    write_output_file(all_recipes)

    # Save manifest
    save_manifest(manifest)

    # Update unrated list
    # Keep existing unrated that are still unrated, add new ones
    existing_unrated_ids = {u["id"] for u in unrated}
    for item in new_unrated:
        if item["id"] not in existing_unrated_ids:
            unrated.append(item)

    # Remove from unrated if they now have ratings
    rated_ids = {r["_id"] for r in all_recipes if r["metadata"]["rating"] is not None}
    unrated = [u for u in unrated if u["id"] not in rated_ids]

    save_unrated(unrated)

    if unrated:
        print(f"\nUnrated recipes ({len(unrated)}):")
        for item in unrated:
            print(f"  - {item['title']} ({item['id']})")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Process raw recipes using BAML for V2 pipeline"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Reprocess all recipes, ignoring cache"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes"
    )

    args = parser.parse_args()
    process_recipes(force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
