---
github_issue: 16
---
# Handoff: Feature 8 (Sparse Search)

## Current status
- Sparse search capability added with a hashed TF-IDF encoder and dense→sparse fallback logic.
- Fixed a bug where `sparse_encoder` was out of scope and caused a NameError during upsert.
- Renamed sparse env var handling for clarity (`SPARSE_MIN_DOC_FREQ` preferred; `SPARSE_MIN_DF` still supported).

## Key changes
- `main.py`
  - Adds sparse encoder creation inside `main(...)` and uses it for upsert + search.
  - Dense-first search; if insufficient dense hits, fallback to sparse; if sparse fails, use dense fallback or LLM.
  - New CLI options: `--sparse-threshold`, `--min-dense-hits`, `--dense-top-k`, `--sparse-top-k`.
  - Env vars: `SPARSE_HASH_DIM`, `SPARSE_MIN_DOC_FREQ` (fallback to `SPARSE_MIN_DF`).
- `utils/sparse_helper.py` (new)
  - Tokenization + hashed TF-IDF encoder with optional bigrams and L2 normalization.
- `utils/response_formatter.py`
  - New `RecipeSource.RAG_SPARSE` with “sparse search” header.
- Tests
  - `tests/test_mvp.py` updated with routing tests and sparse header check.
  - `tests/test_sparse_helper.py` added for encoder behavior.

## Files changed/added
- Modified: `main.py`, `utils/response_formatter.py`, `tests/test_mvp.py`
- Added: `utils/sparse_helper.py`, `tests/test_sparse_helper.py`

## Tests run
- `./venv/bin/python -m pytest -q`
  - Fails during collection due to existing issues:
    - `tests/test_transform_recipes.py`: missing `get_existing_processed_recipes` in `scripts/transform_recipes.py`
    - `v2/tests/test_process_recipes.py`: import error `tests.test_process_recipes`

## Notes
- Manual test requested by user before further changes.
- If adjusting sparse behavior, start with `SPARSE_HASH_DIM` and `SPARSE_MIN_DOC_FREQ`.
- Sparse upsert only attaches `sparse_values` if encoder returns non-empty indices.

## Next steps
- Fix bug (see below)
- Support user’s manual testing feedback.
- Consider tightening sparse thresholds or refining tokenization if too many noisy hits.
- Re-run tests after addressing existing test collection errors.

## Bug
- User ran the following command: `python main.py --query "I'd like a dish that two people can make together in under an hour. I want it to be moderately spicy and have meat in it." --upsert`
  - This caused the following error: ` raise PineconeApiException(http_resp=r)
 pinecone.exceptions.exceptions.PineconeApiException: (400)
 Reason: Bad Request
 HTTP response headers: HTTPHeaderDict({'Date': 'Wed, 28 Jan 2026 03:48:55 GMT', 'Content-Type': 'text/plain; charset=utf-8', 'Content-Length': '139', 'Connection': 'keep-alive', 'x-pinecone-api-version': '2025-10', 'x-envoy-upstream-service-time': '33', 'x-pinecone-response-duration-ms': '34', 'server': 'envoy'})
 HTTP response body: {"error":{"code":"INVALID_ARGUMENT","message":"Invalid query: Dense index queries with 'vector' input must specify 'values'"},"status":400}`
  - I think that somewhere the Pinecone API is not being used correctly. 
  - I did not get this or any other error before I started the session to add the sparse search. 

## Open requests
- User may want clearer naming; confirm if they want CLI flag renames.
- After completion, offer to close CFS document with: `cfs i features complete 8 --force`.
