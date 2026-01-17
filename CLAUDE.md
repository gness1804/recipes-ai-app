# CLAUDE.md

Project-specific instructions for Claude Code when working in this repository.

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
Records in `data/seeds.py` follow this structure:
```python
{
    "_id": "recipe-slug",
    "content": "Full recipe description for embedding",
    "category": "seafood|vegetarian|beef|etc",
    "diet": "pescatarian|gluten-free|vegetarian|etc",
    "prep_time": "20m|30m|1h|etc",
    "difficulty": "easy|medium|hard",
    "rating": "1.0-5.0",
}
```

## Testing

Tests should cover:
- Embedding generation
- Pinecone upsert/query operations
- Recipe matching logic
- Fallback behavior when no matches found
