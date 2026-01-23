# AGENTS.md

Project-specific instructions for Claude Code and other AI agents (Cursor, Gemini, etc) when working in this repository.

## Project Overview

This is a recipe chatbot application that uses RAG (Retrieval-Augmented Generation) with a Pinecone vector database. Users can ask natural language questions like:
- "Give me a good seafood recipe for a weeknight"
- "Give me an easy 30-minute beef recipe with minimal dairy"

The app will search the user's personal recipe collection first, then fall back to generating or web-searching for recipes if no suitable match is found.

## Architecture

### Data Flow
1. User submits a natural language query
2. Query is embedded using OpenAI's embedding model
3. Pinecone performs semantic search + reranking on personal recipes
4. If a good match is found, return that recipe
5. If no match, fall back to LLM generation or web search (not yet implemented)

### Key Components
- `main.py` - Entry point; handles upserts and sample queries
- `utils/embedding_helper.py` - OpenAI embedding utilities
- `data/seeds.py` - Seed recipe records for the vector database
- `data/raw-recipes/` - Source recipe files

### External Services
- **Pinecone** - Vector database for recipe storage and semantic search
- **OpenAI** - Embeddings (text-embedding-3-small) and future chat completion

## Development

### Environment Variables
Required in `.env`:
- `PINECONE_API_KEY` - Pinecone API key
- `PINECONE_INDEX` - Index name (default: `recipes-vector-db`)
- `PINECONE_NAMESPACE` - Namespace (default: `main_recipes`)
- `OPENAI_API_KEY` - OpenAI API key
- `EMBEDDING_MODEL` - Embedding model (default: `text-embedding-3-small`)

### Running the App
```bash
# Activate virtual environment
source venv/bin/activate

# Run with upsert (seeds data into Pinecone)
python main.py

# Run without upserting (query only)
python main.py --skip-upsert
```

### Recipe Record Format
As noted in data/schemas/basicSchema.json`, the RAG ingestion-ready schema for a recipe is:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Recipe Vector DB Schema",
  "description": "Schema for recipe records optimized for RAG vector database ingestion. Enum values are extensible - the LLM classifier may add new values if existing ones don't fit.",
  "type": "object",
  "required": ["_id", "content", "metadata"],
  "properties": {
    "_id": {
      "type": "string",
      "description": "Unique identifier derived from recipe slug (e.g., 'thai-chicken')"
    },
    "content": {
      "type": "string",
      "description": "Full recipe text for embedding generation"
    },
    "metadata": {
      "type": "object",
      "required": ["title"],
      "properties": {
        "title": {
          "type": "string",
          "description": "Recipe title from the H1 header"
        },
        "ingredients": {
          "type": "string",
          "description": "Full ingredients section text (for display, not filtering)"
        },
        "instructions": {
          "type": "string",
          "description": "Full instructions section text (for display, not filtering)"
        },
        "rating": {
          "type": ["number", "null"],
          "minimum": 1,
          "maximum": 10,
          "description": "Rating from 1.0-10.0, or null if unrated"
        },
        "diet": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Dietary classifications. Base values: vegetarian, vegan, pescatarian, dairy-free, gluten-free. LLM may add others.",
          "examples": [["vegetarian", "gluten-free"], ["pescatarian"]]
        },
        "protein": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Protein sources in the recipe. Base values: beef, pork, lamb, chicken, turkey, seafood, tofu, egg, none. LLM may add others.",
          "examples": [["chicken"], ["seafood", "pork"], ["none"]]
        },
        "cuisine": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Cuisine/regional style. Base values: american, cajun, caribbean, chinese, cuban, indian, italian, japanese, korean, mexican, thai, african, brazilian, mediterranean, other. LLM may add others.",
          "examples": [["cajun"], ["thai", "chinese"]]
        },
        "meal_type": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Meal category. Base values: main-dish, side-dish, dessert, sauce, soup, salad, breakfast, snack. LLM may add others.",
          "examples": [["main-dish"], ["side-dish", "snack"]]
        },
        "difficulty": {
          "type": "string",
          "enum": ["easy", "medium", "hard"],
          "description": "Recipe difficulty level"
        },
        "prepTimeMinutes": {
          "type": ["integer", "null"],
          "minimum": 1,
          "description": "Total preparation and cooking time in minutes, or null if not determinable"
        }
      }
    }
  }
}
```

## Testing

Tests should cover:
- Embedding generation
- Pinecone upsert/query operations
- Recipe matching logic
- Fallback behavior when no matches found
