#!/usr/bin/env python3
"""
LLM Classifier Module

Classifies recipes using GPT-4o for metadata fields that require
semantic understanding: diet, protein, cuisine, meal_type, difficulty, prepTimeMinutes.

Supports dynamic enum expansion - if the LLM determines a recipe doesn't fit
existing categories, it can propose new values.
"""

import json
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_FILE = CACHE_DIR / "classifications.json"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Base enum values - LLM can extend these if needed
BASE_ENUMS = {
    "diet": [
        "vegetarian",
        "vegan",
        "pescatarian",
        "dairy-free",
        "gluten-free",
    ],
    "protein": [
        "beef",
        "pork",
        "lamb",
        "chicken",
        "turkey",
        "seafood",
        "tofu",
        "egg",
        "none",
    ],
    "cuisine": [
        "american",
        "cajun",
        "caribbean",
        "chinese",
        "cuban",
        "indian",
        "italian",
        "japanese",
        "korean",
        "mexican",
        "thai",
        "african",
        "brazilian",
        "mediterranean",
        "other",
    ],
    "meal_type": [
        "main-dish",
        "side-dish",
        "dessert",
        "sauce",
        "soup",
        "salad",
        "breakfast",
        "snack",
    ],
    "difficulty": ["easy", "medium", "hard"],
}

# Track discovered enum values across all classifications
discovered_enums: dict[str, set[str]] = {key: set() for key in BASE_ENUMS}


class ClassificationResult(TypedDict):
    diet: list[str]
    protein: list[str]
    cuisine: list[str]
    meal_type: list[str]
    difficulty: str
    prepTimeMinutes: int | None


SYSTEM_PROMPT = """You are a recipe classification assistant. Given a recipe, you will classify it according to specific metadata fields.

## Fields to Classify

1. **diet** (array): Dietary restrictions this recipe satisfies.
   - Base values: vegetarian, vegan, pescatarian, dairy-free, gluten-free
   - A recipe can have multiple values (e.g., ["vegetarian", "gluten-free"])
   - If it doesn't fit any dietary restriction, return an empty array []
   - You MAY add new values if the recipe clearly fits a dietary category not listed (e.g., "keto", "paleo")

2. **protein** (array): Protein sources in the recipe.
   - Base values: beef, pork, lamb, chicken, turkey, seafood, tofu, egg, none
   - "seafood" covers fish, shrimp, clams, etc.
   - Use "none" for recipes with no significant protein source
   - A recipe can have multiple proteins (e.g., jambalaya with ["chicken", "seafood", "pork"])
   - You MAY add new values if needed (e.g., "duck", "game")

3. **cuisine** (array): Cuisine or regional style.
   - Base values: american, cajun, caribbean, chinese, cuban, indian, italian, japanese, korean, mexican, thai, african, brazilian, mediterranean, other
   - Use "other" sparingly - prefer adding a specific cuisine if identifiable
   - A recipe can blend cuisines (e.g., ["thai", "chinese"] for a fusion dish)
   - You MAY add new values (e.g., "indonesian", "vietnamese", "spanish")

4. **meal_type** (array): What category of meal this is.
   - Base values: main-dish, side-dish, dessert, sauce, soup, salad, breakfast, snack
   - A recipe can be multiple types (e.g., a substantial salad could be ["salad", "main-dish"])
   - You MAY add new values if needed (e.g., "appetizer", "beverage", "condiment")

5. **difficulty** (string): How difficult the recipe is.
   - Values: easy, medium, hard
   - easy: Few ingredients, simple techniques, under 30 mins active time
   - medium: More ingredients or techniques, 30-60 mins active time
   - hard: Complex techniques, many steps, over 60 mins active time, or requires special skills

6. **prepTimeMinutes** (integer or null): Total time to prepare and cook the recipe.
   - Estimate based on the instructions
   - Include both prep time and cooking/baking time
   - Return null if you cannot reasonably estimate

## Rules for Adding New Values

Before adding a new enum value:
1. Check if an existing value already covers it (e.g., don't add "shrimp" when "seafood" exists)
2. Use lowercase with hyphens for multi-word values (e.g., "dairy-free", "main-dish")
3. Be specific but not overly narrow (e.g., "vietnamese" is good, "hanoi-style" is too narrow)

## Output Format

Return ONLY valid JSON with this structure:
{
  "diet": [],
  "protein": [],
  "cuisine": [],
  "meal_type": [],
  "difficulty": "easy|medium|hard",
  "prepTimeMinutes": null or integer
}

Do not include any explanation or text outside the JSON."""


def load_cache() -> dict[str, ClassificationResult]:
    """Load cached classifications from disk."""
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict[str, ClassificationResult]) -> None:
    """Save classifications cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8"
    )


def get_all_known_values(field: str) -> list[str]:
    """Get all known values for a field (base + discovered)."""
    base = BASE_ENUMS.get(field, [])
    discovered = discovered_enums.get(field, set())
    return sorted(set(base) | discovered)


def classify_recipe(
    recipe_id: str,
    recipe_content: str,
    use_cache: bool = True,
) -> ClassificationResult:
    """
    Classify a recipe using GPT-4o.

    Args:
        recipe_id: Unique identifier for the recipe (used for caching)
        recipe_content: Full recipe text to classify
        use_cache: Whether to use cached results if available

    Returns:
        ClassificationResult with all classified fields
    """
    # Check cache first
    if use_cache:
        cache = load_cache()
        if recipe_id in cache:
            return cache[recipe_id]

    # Build user message with current known values
    known_values_info = "\n".join(
        f"- {field}: {', '.join(get_all_known_values(field))}"
        for field in ["diet", "protein", "cuisine", "meal_type"]
    )

    user_message = f"""Currently known values (you may add new ones if needed):
{known_values_info}

Classify this recipe:

{recipe_content}"""

    # Call GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=500,
    )

    # Parse response
    response_text = response.choices[0].message.content.strip()

    # Handle markdown code fences if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        # Remove first line (```json) and last line (```)
        response_text = "\n".join(lines[1:-1])

    result: ClassificationResult = json.loads(response_text)

    # Track any new values discovered
    for field in ["diet", "protein", "cuisine", "meal_type"]:
        for value in result.get(field, []):
            if value not in BASE_ENUMS.get(field, []):
                discovered_enums[field].add(value)

    # Save to cache
    if use_cache:
        cache = load_cache()
        cache[recipe_id] = result
        save_cache(cache)

    return result


def classify_recipes_batch(
    recipes: list[tuple[str, str]],
    use_cache: bool = True,
    progress_callback: callable = None,
) -> dict[str, ClassificationResult]:
    """
    Classify multiple recipes.

    Args:
        recipes: List of (recipe_id, recipe_content) tuples
        use_cache: Whether to use cached results
        progress_callback: Optional callback(current, total, recipe_id) for progress reporting

    Returns:
        Dict mapping recipe_id to ClassificationResult
    """
    results = {}
    total = len(recipes)

    for i, (recipe_id, content) in enumerate(recipes, 1):
        if progress_callback:
            progress_callback(i, total, recipe_id)

        results[recipe_id] = classify_recipe(recipe_id, content, use_cache)

    return results


def get_discovered_enums() -> dict[str, list[str]]:
    """Get all newly discovered enum values (not in base enums)."""
    return {
        field: sorted(values)
        for field, values in discovered_enums.items()
        if values
    }


def clear_cache() -> None:
    """Clear the classifications cache."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()


if __name__ == "__main__":
    # Simple test
    test_recipe = """# Thai Chicken

Rating: 7/10

## Ingredients
- 2.5 cups instant rice
- 4 skinless, boneless chicken breast halves
- Spicy Peanut Sauce
- 2 teaspoons vegetable oil
- 1 tablespoon bottled minced garlic

## Instructions
1. Bring 2.5 cups of water to a boil. Add the rice, cover, and remove from heat.
2. Heat the oil in a 12-inch nonstick skillet over high heat.
3. Cut the chicken into strips, cook until no longer pink.
4. Add sauce and serve over rice.
"""

    print("Testing LLM classifier...")
    result = classify_recipe("test-thai-chicken", test_recipe, use_cache=False)
    print(json.dumps(result, indent=2))
