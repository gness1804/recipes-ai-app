# Recipe Chatbot

A personal recipe chatbot that uses RAG (Retrieval-Augmented Generation) with a Pinecone vector database. Ask natural-language questions like "Give me a good weeknight seafood recipe" and the app searches your recipe collection, falling back to LLM generation when no suitable match is found.

https://recipes-ai-app.onrender.com/

<img width="1461" height="709" alt="Recipe Chatbot screenshot" src="https://github.com/user-attachments/assets/63c15986-1d33-428c-99b0-27a60c0c7fee" />

## Running the App

### Web UI (Streamlit)

```bash
source venv/bin/activate && streamlit run app.py
```

The app will open at `http://localhost:8501`. Enter your OpenAI API key in the sidebar — it is Fernet-encrypted and stored as a browser cookie for 7 days.

**Owner vs. guest behavior:**
- **Owner** (your `OWNER_OPENAI_API_KEY`): Full RAG pipeline — searches your personal Pinecone recipe collection, falls back to LLM generation only when no match is found.
- **Other users**: LLM-only recipe generation — no access to your personal collection or Pinecone account.

### CLI (interactive mode)

```bash
source venv/bin/activate && python main.py
```

See `python main.py --help` for all options (`--upsert`, `--dense-only`, `--threshold`, etc.).

## Environment Variables

Create a `.env` file in the project root:

```env
# Required for all modes
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key

# Optional — defaults shown
PINECONE_INDEX=recipes-vector-db
PINECONE_NAMESPACE=recipes
EMBEDDING_MODEL=text-embedding-3-small
SPARSE_HASH_DIM=262144
SPARSE_MIN_DOC_FREQ=1
# Legacy alias also supported in CLI + web UI:
# SPARSE_MIN_DF=1
MATCH_THRESHOLD=0.10
SPARSE_THRESHOLD=0.0
MIN_DENSE_HITS=3
DENSE_TOP_K=10
SPARSE_TOP_K=10
SEARCH_DIAGNOSTICS=0

# Web UI — required for persistent encrypted cookies
SESSION_SECRET=   # generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Web UI — owner detection (users with this key get full RAG access)
OWNER_OPENAI_API_KEY=your-openai-api-key
```

`main.py` now uses these env vars as its CLI defaults, so web and CLI search parameters stay in sync unless you explicitly override flags.
By default, non-`--dense-only` queries run both dense and sparse retrieval and select the stronger passing result.
Set `SEARCH_DIAGNOSTICS=1` to append route/score diagnostics to responses in the web UI.

## Deployment (Render)

Deployed on [Render](https://render.com) using the Starter plan ($7/mo, always-on, full WebSocket support). The runtime image is built from the project `Dockerfile`, which intentionally excludes offline-only modules (`baml_client/`, `baml_src/`, `v2/`, `scripts/`, `tests/`, raw recipe sources) to keep the image small.

1. Create a new **Web Service** on Render and connect this repo.
2. Set **Runtime** to **Docker** — do not use the auto-detected "Python 3" runtime, which would ignore the Dockerfile.
3. Choose the **Starter** plan (always-on; cold starts are not acceptable for a chat UI).
4. Set the tracked branch to `master`.
5. In **Environment**, add the variables listed below. Generate a fresh `SESSION_SECRET` for production — do not reuse any development value.
6. Deploy. The app is served on port `8501` and reachable at `https://<service-name>.onrender.com`.

### Required environment variables

- `PINECONE_API_KEY` — Pinecone API key
- `OPENAI_API_KEY` — OpenAI API key
- `OWNER_OPENAI_API_KEY` — owner key; users who paste this key in the sidebar get full RAG access
- `SESSION_SECRET` — fresh Fernet key (`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`)

### Recommended tuning variables

```env
PINECONE_INDEX=recipes-vector-db
PINECONE_NAMESPACE=recipes
EMBEDDING_MODEL=text-embedding-3-small
SPARSE_HASH_DIM=262144
SPARSE_MIN_DOC_FREQ=1
MATCH_THRESHOLD=0.10
SPARSE_THRESHOLD=0.0
MIN_DENSE_HITS=3
DENSE_TOP_K=10
SPARSE_TOP_K=10
# SEARCH_DIAGNOSTICS=1   # optional — appends route/score diagnostics to responses
```

`PINECONE_NAMESPACE` must match the actual namespace in your Pinecone index. A mismatch silently returns zero hits and falls back to LLM generation.

### Custom domain

Render supports custom domains directly on the Starter plan via dashboard → Settings → Custom Domains. None is configured for this project; the default `*.onrender.com` URL is used.

## Installing Dependencies

```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

Or, to install only web dependencies:

```bash
pip install ".[web]"
```

## Adding New Recipes & Seeding the Database

To add a new raw recipe (such as a Word document, PDF, or image) and sync it with the database, follow the v2 ingestion pipeline:

1. **Prepare the file**: Ensure your recipe is in a supported format: PDF, image (`.jpg`, `.png`, etc.), or text (`.txt`, `.md`). *(Note: Word documents like `.doc` or `.docx` must be saved/exported as PDF or text first).*
2. **Place the file**: Move the raw recipe file into the `v2/raw-recipes/` directory.
3. **Process with BAML**: Extract structured data from the raw file using the v2 processing script. This updates `v2/recipes_for_vector_db.py`:
   ```bash
   python v2/process_recipes.py
   ```
4. **Combine Datasets**: Merge the newly processed v2 recipes with the existing dataset to create a unified `data/combined_recipes.py` file:
   ```bash
   python scripts/combine_recipe_datasets.py
   ```
5. **Upsert to Pinecone**: Embed the combined recipes and push them to your Pinecone vector database:
   ```bash
   python main.py --upsert
   ```

## Architecture

- `app.py` — Streamlit frontend
- `session.py` — Fernet API key encryption and owner detection
- `main.py` — CLI entry point; `process_query()` is called directly by the web UI
- `utils/embedding_helper.py` — OpenAI embedding utilities
- `utils/llm_helper.py` — Recipe response generation and fallback
- `utils/sparse_helper.py` — BM25-style sparse encoder for hybrid search
- `data/` — Recipe seed data and combined records
