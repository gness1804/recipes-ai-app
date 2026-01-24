# Handoff: V2 Recipe Processing with BAML

**Date:** 2026-01-23
**Branch:** `refactor/baml-for-recipes`
**Version:** 0.2.0

## Summary

Implemented a new recipe processing pipeline for v2 that uses BAML to transform raw recipes (images or text) into structured data suitable for RAG vector database ingestion. The LLM now automatically rates recipes based on dietary preferences.

## What Was Completed

### GitHub Issues Closed
- **Issue #6**: Data Transformation Version 2 Use BAML
- **Issue #8**: Make Rating Determined By LLM Rather Than Manually Entered

### Key Changes

1. **Updated BAML Schema** (`baml_src/recipe.baml`)
   - Embedded dietary preferences directly in the `rating` field description
   - LLM now evaluates recipes against user's preferences and assigns ratings 1-10

2. **Created Processing Script** (`v2/process_recipes.py`)
   - Processes images via `ExtractRecipeFromImage` (gpt-4o-mini)
   - Processes text via `ExtractRecipe` (gpt-5-mini)
   - Transforms BAML output to `basicSchema.json` format
   - Incremental processing (skips already-processed recipes)
   - CLI flags: `--force`, `--dry-run`

3. **Output Files**
   - `v2/recipes_for_vector_db.py` - Processed recipes (matches v1 format)
   - `v2/_processed_manifest.json` - Tracks which files have been processed
   - `v2/_unrated.json` - Lists recipes where LLM returned null rating

4. **Tests** (`v2/tests/test_process_recipes.py`)
   - 12 passing tests covering slugify, transformation, and formatting

5. **Version Management**
   - Added `pyproject.toml` with bump2version config
   - Added `.bumpversion.cfg`
   - Current version: 0.2.0, tagged as `v0.2.0`

6. **Gitignore**
   - Added `v2/raw-recipes/` to keep source files local

## Usage

```bash
# Process new recipes only
python v2/process_recipes.py

# Force reprocess all
python v2/process_recipes.py --force

# Preview what would be processed
python v2/process_recipes.py --dry-run

# Bump version
bump2version patch|minor|major
```

## File Structure

```
v2/
├── raw-recipes/                    # Input: raw recipe files (gitignored)
├── recipes_for_vector_db.py        # Output: processed recipes for RAG
├── _processed_manifest.json        # Tracking: processed files
├── _unrated.json                   # Tracking: recipes needing manual rating
├── process_recipes.py              # Script: main processing logic
├── dietary_preferences.md          # Reference: rating criteria
└── tests/
    └── test_process_recipes.py     # Tests
```

## Next Steps / Future Work

- Add more raw recipes to `v2/raw-recipes/` and run processing
- Integrate v2 recipes into the main RAG ingestion pipeline
- Consider adding a combined script that merges v1 and v2 recipes for ingestion
- May want to add support for PDF recipes in v2

## Commits

```
5238fe8 Bump version: 0.1.0 → 0.2.0
cd0de9f chore: Add bump2version setup and update .gitignore
fb75203 feat: Add v2 recipe processing pipeline with BAML and LLM-based ratings
```
