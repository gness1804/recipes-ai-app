---
github_issue: 29
---
# Add startup namespace vector-count safety check in Streamlit app

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents

## Summary
Add a startup safety check in the Streamlit owner path to detect misconfigured Pinecone namespace/index early.

## Context
We had a deployment where retrieval silently returned zero dense/sparse hits because PINECONE_NAMESPACE in deployment did not match the namespace containing vectors. The app then fell back to LLM generation, which masked the root cause.

## Proposed behavior
- In _get_pinecone_resources() (cached), call index.describe_index_stats() after initializing the index.
- Read vector count for configured PINECONE_NAMESPACE.
- If namespace is missing or has zero vectors:
  - Log a clear warning including index + namespace.
  - Surface a visible owner-facing warning in the UI that retrieval is likely misconfigured and LLM fallback may occur.
- If namespace has vectors, continue normally with no warning.

## Optional strict mode
- Add env toggle STRICT_NAMESPACE_CHECK=1 to fail fast (raise error) instead of warning when namespace count is zero.
- Default should remain non-breaking (warning only).


## Acceptance criteria

- Misconfigured/empty namespace is clearly surfaced at startup (not only after confusing query output).
- Normal path remains unchanged when namespace has vectors.
- Add tests for both warning mode and strict mode behavior.
- Document new behavior and env toggle in README.

<!-- DONE -->
