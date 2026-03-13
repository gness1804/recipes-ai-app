# Recipe Chatbot

A personal recipe chatbot that uses RAG (Retrieval-Augmented Generation) with a Pinecone vector database. Ask natural-language questions like "Give me a good weeknight seafood recipe" and the app searches your recipe collection, falling back to LLM generation when no suitable match is found.

https://recipes-ai-chatbot.streamlit.app/

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
PINECONE_NAMESPACE=main_recipes
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

## Deployment (Streamlit Community Cloud)

1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Select this repository and branch.
4. Set the app entrypoint to `app.py`.
5. In **App settings -> Secrets**, paste values from `.streamlit/secrets.toml.example` (replace placeholders).
6. Deploy the app.

If you want to validate locally with Streamlit-style secrets first:

```bash
mkdir -p .streamlit && cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then fill in real keys and run:

```bash
source venv/bin/activate && streamlit run app.py
```

### Deployment Troubleshooting

- Error: `No matching distribution found for streamlit-cookies-controller>=0.3.0`
  - Fix: use `streamlit-cookies-controller>=0.0.4`.
- Error from Pinecone package import saying `pinecone-client` was renamed
  - Fix: replace `pinecone-client` with `pinecone` in dependencies.

Current expected dependency lines:

```txt
pinecone>=3.0.0
streamlit-cookies-controller>=0.0.4
```

### Known Fixes (Streamlit Cloud)

- `KeyError: 'utils.llm_helper'` at app startup
  - Fix shipped: add `utils/__init__.py` so `utils` is always treated as a package in cloud runtime.
- First preset query button after redeploy did nothing
  - Fix shipped: preserve `pending_query` until API key is available, then consume it.

### Custom URL Notes

As of March 2, 2026, Streamlit Community Cloud hosts apps on the `*.streamlit.app` domain.  
That means you can choose a custom app subdomain like:

- `recipes-ai-chatbot.streamlit.app`

For a true custom domain like `recipes-ai-chatbot.com`, Streamlit Community Cloud does not provide direct custom-domain mapping. Your options are:

1. Keep Streamlit hosting and use a redirect/proxy layer in front (for example via Cloudflare), with the caveat that this is outside Streamlit's native setup.
2. Deploy on infrastructure where you control the web server/domain mapping directly, then point Cloudflare DNS there.

For this project, the fastest path to satisfy a custom URL requirement is using a custom Streamlit subdomain (`recipes-ai-chatbot.streamlit.app`).

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
