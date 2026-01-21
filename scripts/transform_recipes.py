#!/usr/bin/env python3
"""
Recipe Transformer Script

Transforms raw recipes from various formats (Markdown, PDF, images) into a
standardized Markdown format optimized for RAG ingestion.

Usage:
    python scripts/transform_recipes.py              # Process all (with confirmation)
    python scripts/transform_recipes.py --limit 5   # Process first 5 recipes
    python scripts/transform_recipes.py --yes       # Skip confirmation
    python scripts/transform_recipes.py -d          # Allow reprocessing duplicates
"""

import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image

# Load environment variables
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_RECIPES_DIR = PROJECT_ROOT / "data" / "raw-recipes"
PROCESSED_RECIPES_DIR = PROJECT_ROOT / "data" / "processed-recipes"
MISSING_RATINGS_FILE = PROCESSED_RECIPES_DIR / "_missing_ratings.md"
MANIFEST_FILE = PROCESSED_RECIPES_DIR / "_processed_manifest.json"

# Supported file extensions
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".txt"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a recipe formatting assistant. Your job is to transform raw recipe content into a standardized Markdown format.

## Output Format

You MUST output the recipe in exactly this format:

```
# [Recipe Name - use the provided original title]

Rating: [N]/10

## Ingredients

[If recipe has multiple parts, use H3 headers like "### For the Sauce"]
- [quantity] [unit] [ingredient], [preparation notes if any]
- ...

## Instructions

[If recipe has multiple parts, use H3 headers like "### For the Sauce"]
1. [Step 1]
2. [Step 2]
...
```

## Formatting Rules

### Quantities
- Convert ALL fractions to decimals: 1/2 → 0.5, 1/4 → 0.25, 3/4 → 0.75, 1/3 → 0.33, 2/3 → 0.67
- Normalize units:
  - "T", "Tbsp", "tbsp" → "tablespoons" (or "tablespoon" if singular)
  - "t", "tsp" → "teaspoons" (or "teaspoon" if singular)
  - "c", "C" → "cup" or "cups"
  - "oz" → "oz."
  - "lb", "lbs" → "lb." or "lbs."
- Use consistent formatting: "2 tablespoons olive oil" not "2 T olive oil"

### Compound Ingredients (canned goods, etc.)
- Format: "[quantity] [container] ([size]) [ingredient], [preparation]"
- Examples:
  - "1 (15 ounce) can chickpeas" → "1 can (15 oz.) chickpeas"
  - "2 (14.5 ounce) cans diced tomatoes" → "2 cans (14.5 oz. ea.) diced tomatoes"

### Ingredient Details
INCLUDE:
- Quantities (normalized as above)
- Preparation notes: "chopped", "diced", "minced", "cut into 1-inch cubes"
- Essential descriptors: "with juice", "drained", "room temperature", "softened"
- Optional markers: "(optional)" at the end

EXCLUDE:
- Calorie counts or nutritional information like "(240 cal, 0g protein)"
- Editorial comments like "use the good stuff" or "I prefer brand X"
- Recipe totals or summaries

### Multi-Part Recipes
If a recipe has distinct parts (e.g., a stir-fry and its sauce), use H3 headers:
```
## Ingredients

### For the Stir-Fry
- 1 lb. chicken breast, cut into 1-inch cubes
...

### For the Peanut Sauce
- 0.5 cup peanut butter
...

## Instructions

### For the Stir-Fry
1. Heat oil in wok...

### For the Peanut Sauce
1. Combine ingredients...
```

### Recipe Title (H1)
- IMPORTANT: The user will provide the original recipe title. Use that title for the H1 header.
- Clean up the title if needed (remove file extensions, fix obvious typos), but preserve the recipe name.
- Do NOT rename the recipe based on the content - use the provided title.

### Rating
- If a rating exists in the original, normalize it to X/10 format
- Convert star ratings: "4 out of 5 stars" → "8/10"
- If NO rating exists, output exactly: "Rating: [MISSING]"

### What to EXCLUDE from output
- URLs or source links
- Editorial commentary or preambles ("This is a great recipe I got from...")
- "Total ingredients: X" or "Total steps: X" counts
- Nutritional information
- Personal notes unrelated to cooking

Output ONLY the formatted recipe. No explanations or additional text."""


def slugify(name: str) -> str:
    """Convert a recipe name to a filename-safe slug."""
    # Remove special characters, convert to lowercase, replace spaces with hyphens
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def get_raw_recipes() -> list[Path]:
    """Get all recipe files from the raw recipes directory."""
    recipes = []
    for ext in MARKDOWN_EXTENSIONS | PDF_EXTENSIONS | IMAGE_EXTENSIONS:
        recipes.extend(RAW_RECIPES_DIR.glob(f"*{ext}"))
    return sorted(recipes)


def load_manifest() -> dict[str, str]:
    """Load the processed recipes manifest.

    Returns:
        dict mapping raw filenames to processed filenames
    """
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict[str, str]) -> None:
    """Save the processed recipes manifest."""
    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pil_image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to base64."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_image_media_type(path: Path) -> str:
    """Get the media type for an image file."""
    ext = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(ext, "image/png")


def call_gpt4o_text(content: str, original_title: str) -> str:
    """Call GPT-4o with text content."""
    user_message = f"Original recipe title: {original_title}\n\nTransform this recipe:\n\n{content}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=2000,
    )
    return response.choices[0].message.content


def call_gpt4o_vision(
    images: list[str], original_title: str, media_type: str = "image/png"
) -> str:
    """Call GPT-4o with image content."""
    image_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{img}",
                "detail": "high",
            },
        }
        for img in images
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Original recipe title: {original_title}\n\nExtract and transform the recipe from this image into the standardized format:",
                    },
                    *image_content,
                ],
            },
        ],
        temperature=0.1,
        max_tokens=2000,
    )
    return response.choices[0].message.content


def process_markdown(file_path: Path, original_title: str) -> str:
    """Process a markdown file."""
    content = file_path.read_text(encoding="utf-8")
    return call_gpt4o_text(content, original_title)


def process_pdf(file_path: Path, original_title: str) -> str:
    """Process a PDF file by converting pages to images."""
    # Convert PDF pages to images
    images = convert_from_path(file_path, dpi=150)

    # Convert PIL images to base64
    base64_images = [pil_image_to_base64(img) for img in images]

    return call_gpt4o_vision(base64_images, original_title)


def process_image(file_path: Path, original_title: str) -> str:
    """Process an image file."""
    base64_image = encode_image_to_base64(file_path)
    media_type = get_image_media_type(file_path)
    return call_gpt4o_vision([base64_image], original_title, media_type)


def process_recipe(file_path: Path) -> tuple[str, bool]:
    """
    Process a recipe file and return the transformed content.

    Returns:
        tuple: (transformed_content, has_missing_rating)
    """
    ext = file_path.suffix.lower()
    # Extract original title from filename (without extension)
    original_title = file_path.stem

    if ext in MARKDOWN_EXTENSIONS:
        result = process_markdown(file_path, original_title)
    elif ext in PDF_EXTENSIONS:
        result = process_pdf(file_path, original_title)
    elif ext in IMAGE_EXTENSIONS:
        result = process_image(file_path, original_title)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Strip markdown code fences if present
    result = strip_markdown_fences(result)

    # Check for missing rating
    has_missing_rating = "[MISSING]" in result

    return result, has_missing_rating


def strip_markdown_fences(content: str) -> str:
    """Remove markdown code fences if GPT-4o wrapped the output in them."""
    content = content.strip()
    # Remove opening fence (```markdown or ```)
    if content.startswith("```"):
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1 :]
    # Remove closing fence
    if content.endswith("```"):
        content = content[:-3].rstrip()
    return content


def extract_recipe_name(content: str) -> str:
    """Extract the recipe name from the transformed content."""
    # Look for the H1 header
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "unknown-recipe"


def save_processed_recipe(content: str, original_path: Path) -> Path:
    """Save the processed recipe to the output directory."""
    # Use the original filename for the slug to preserve user's naming
    slug = slugify(original_path.stem)

    # Ensure unique filename
    output_path = PROCESSED_RECIPES_DIR / f"{slug}.md"

    output_path.write_text(content, encoding="utf-8")
    return output_path


def load_existing_missing_ratings() -> dict[str, str]:
    """Load existing missing ratings from the report file.

    Returns:
        dict mapping filename to recipe name
    """
    if not MISSING_RATINGS_FILE.exists():
        return {}

    existing = {}
    content = MISSING_RATINGS_FILE.read_text(encoding="utf-8")

    # Parse lines like: - [ ] Recipe Name (`filename.md`)
    for line in content.splitlines():
        if line.startswith("- [ ]"):
            # Extract recipe name and filename
            match = re.match(r"- \[ \] (.+) \(`(.+)`\)", line)
            if match:
                recipe_name, filename = match.groups()
                existing[filename] = recipe_name

    return existing


def scan_processed_recipes_for_missing_ratings() -> list[tuple[str, str]]:
    """Scan all processed recipes and find those missing ratings.

    Returns:
        List of (recipe_name, filename) tuples for recipes missing ratings
    """
    missing = []

    for file in sorted(PROCESSED_RECIPES_DIR.glob("*.md")):
        # Skip special files
        if file.name.startswith("_"):
            continue

        content = file.read_text(encoding="utf-8")

        # Extract recipe name from H1 header
        recipe_name = file.stem.replace("-", " ").title()
        for line in content.splitlines():
            if line.startswith("# "):
                recipe_name = line[2:].strip()
                break

        # Check for missing rating
        has_missing_rating = False
        has_rating_field = False

        for line in content.splitlines():
            if line.startswith("Rating:"):
                has_rating_field = True
                if "[MISSING]" in line:
                    has_missing_rating = True
                break

        # Missing if explicitly marked or no rating field at all
        if has_missing_rating or not has_rating_field:
            missing.append((recipe_name, file.name))

    return missing


def generate_missing_ratings_report(
    new_missing: list[tuple[str, str]],
    now_have_ratings: list[str],
) -> None:
    """Generate a report of recipes missing ratings.

    Args:
        new_missing: List of (recipe_name, filename) tuples for newly processed
            recipes that are missing ratings
        now_have_ratings: List of filenames for recipes that were reprocessed
            and now have ratings (to remove from the report)
    """
    # Load existing entries
    existing = load_existing_missing_ratings()

    # Remove recipes that now have ratings
    for filename in now_have_ratings:
        existing.pop(filename, None)

    # Add new missing ratings
    for recipe_name, filename in new_missing:
        existing[filename] = recipe_name

    # Filter out entries for files that no longer exist (handles renamed/deleted files)
    existing = {
        filename: recipe_name
        for filename, recipe_name in existing.items()
        if (PROCESSED_RECIPES_DIR / filename).exists()
    }

    # If nothing is missing, remove the file
    if not existing:
        if MISSING_RATINGS_FILE.exists():
            MISSING_RATINGS_FILE.unlink()
        return

    # Write combined report
    content = "# Recipes Missing Ratings\n\n"
    content += "Please add ratings to the following recipes:\n\n"

    for filename in sorted(existing.keys()):
        recipe_name = existing[filename]
        content += f"- [ ] {recipe_name} (`{filename}`)\n"

    MISSING_RATINGS_FILE.write_text(content, encoding="utf-8")


def normalize_filenames() -> list[tuple[str, str]]:
    """Normalize processed recipe filenames to match their H1 header slugs.

    Returns:
        List of (old_filename, new_filename) tuples for files that were renamed.
    """
    renamed = []
    manifest = load_manifest()
    manifest_updated = False

    for file in sorted(PROCESSED_RECIPES_DIR.glob("*.md")):
        # Skip special files
        if file.name.startswith("_"):
            continue

        content = file.read_text(encoding="utf-8")

        # Extract recipe name from H1 header
        recipe_name = extract_recipe_name(content)
        if recipe_name == "unknown-recipe":
            print(f"  Warning: Could not extract H1 from {file.name}, skipping")
            continue

        # Generate expected slug
        expected_slug = slugify(recipe_name)
        expected_filename = f"{expected_slug}.md"

        # Check if rename is needed
        if file.name != expected_filename:
            new_path = PROCESSED_RECIPES_DIR / expected_filename

            # Check for conflicts
            if new_path.exists() and new_path != file:
                print(f"  Warning: Cannot rename {file.name} -> {expected_filename} (file exists)")
                continue

            # Rename the file
            file.rename(new_path)
            renamed.append((file.name, expected_filename))

            # Update manifest entries that point to the old filename
            for raw_name, processed_name in list(manifest.items()):
                if processed_name == file.name:
                    manifest[raw_name] = expected_filename
                    manifest_updated = True

    # Save updated manifest
    if manifest_updated:
        save_manifest(manifest)

    return renamed


def main():
    parser = argparse.ArgumentParser(
        description="Transform raw recipes into standardized format for RAG ingestion."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of recipes to process (default: all)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt for batch processing",
    )
    parser.add_argument(
        "--duplicates",
        "-d",
        action="store_true",
        help="Allow reprocessing of already processed recipes",
    )
    parser.add_argument(
        "--scan-ratings",
        action="store_true",
        help="Scan all processed recipes for missing ratings and update the report",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize filenames by renaming files to match their H1 header slugs",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    PROCESSED_RECIPES_DIR.mkdir(parents=True, exist_ok=True)

    # Handle --scan-ratings flag
    if args.scan_ratings:
        print("Scanning processed recipes for missing ratings...")
        missing = scan_processed_recipes_for_missing_ratings()

        if missing:
            # Write the report (replaces existing content entirely)
            content = "# Recipes Missing Ratings\n\n"
            content += "Please add ratings to the following recipes:\n\n"
            for recipe_name, filename in missing:
                content += f"- [ ] {recipe_name} (`{filename}`)\n"
            MISSING_RATINGS_FILE.write_text(content, encoding="utf-8")

            print(f"Found {len(missing)} recipes missing ratings.")
            print(f"Report saved to: {MISSING_RATINGS_FILE}")
        else:
            if MISSING_RATINGS_FILE.exists():
                MISSING_RATINGS_FILE.unlink()
            print("All recipes have ratings!")

        sys.exit(0)

    # Handle --normalize flag
    if args.normalize:
        print("Normalizing processed recipe filenames...")
        renamed = normalize_filenames()

        if renamed:
            print(f"\nRenamed {len(renamed)} files:")
            for old_name, new_name in renamed:
                print(f"  {old_name} -> {new_name}")
            print(f"\nManifest updated: {MANIFEST_FILE}")
        else:
            print("All filenames are already normalized.")

        sys.exit(0)

    # Get all raw recipes
    raw_recipes = get_raw_recipes()

    if not raw_recipes:
        print("No raw recipes found in", RAW_RECIPES_DIR)
        sys.exit(1)

    # Load manifest of already processed recipes
    manifest = load_manifest()

    # Filter out duplicates if not allowed
    recipes_to_process = []
    skipped_duplicates = []

    for recipe_path in raw_recipes:
        if recipe_path.name in manifest and not args.duplicates:
            skipped_duplicates.append(recipe_path)
        else:
            recipes_to_process.append(recipe_path)

    # Apply limit
    if args.limit:
        recipes_to_process = recipes_to_process[: args.limit]

    # Report skipped duplicates
    for recipe_path in skipped_duplicates:
        print(
            f'Skipping "{recipe_path.stem}" - already processed. '
            "Use --duplicates to reprocess."
        )

    if not recipes_to_process:
        print("\nNo new recipes to process.")
        sys.exit(0)

    # Confirmation prompt for processing all
    if args.limit is None and not args.yes:
        print(f"\nAbout to process {len(recipes_to_process)} recipes.")
        # Print all the recipes to be processed just by name. 
        for recipe_path in recipes_to_process:
            print(recipe_path.name)
        print("This will make API calls to OpenAI GPT-4o.\n")
        confirm = input("Continue? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    # Process recipes
    processed_count = 0
    failed_count = 0
    missing_ratings = []
    now_have_ratings = []  # Track recipes that were reprocessed and now have ratings

    print(f"\nProcessing {len(recipes_to_process)} recipes...\n")

    for i, recipe_path in enumerate(recipes_to_process, 1):
        print(
            f"[{i}/{len(recipes_to_process)}] Processing: {recipe_path.name}...",
            end=" ",
        )

        try:
            content, has_missing_rating = process_recipe(recipe_path)
            output_path = save_processed_recipe(content, recipe_path)

            recipe_name = extract_recipe_name(content)

            if has_missing_rating:
                missing_ratings.append((recipe_name, output_path.name))
                print(f"Done (MISSING RATING) -> {output_path.name}")
            else:
                # Track this so we can remove it from missing ratings if it was there before
                now_have_ratings.append(output_path.name)
                print(f"Done -> {output_path.name}")

            # Update manifest with successful processing
            manifest[recipe_path.name] = output_path.name
            save_manifest(manifest)

            processed_count += 1

        except Exception as e:
            print(f"FAILED: {e}")
            failed_count += 1

    # Generate missing ratings report
    generate_missing_ratings_report(missing_ratings, now_have_ratings)

    # Summary
    print(f"\n{'=' * 50}")
    print("Summary:")
    print(f"  Processed: {processed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Skipped (duplicates): {len(skipped_duplicates)}")

    if missing_ratings:
        print(f"\n  Recipes missing ratings: {len(missing_ratings)}")
        print(f"  See: {MISSING_RATINGS_FILE}")

    print(f"\nOutput directory: {PROCESSED_RECIPES_DIR}")


if __name__ == "__main__":
    main()
