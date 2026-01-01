# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

This is a recipe storage and semantic search application built with Pinecone as the vector database backend. The app is in early development stages and currently serves as a template with comprehensive Pinecone agent documentation.

## Pinecone Agent Integration

**MANDATORY**: This repository contains critical Pinecone agent documentation in the `.agents/` directory. Before working on Pinecone-related tasks:

1. **Read `.agents/PINECONE.md`** first to understand:
   - Language selection based on project files
   - Index creation options (CLI, web console, SDK, auto-init)
   - Data operations (upserting records with namespaces, searching, metadata filtering)
   - Critical constraints (batch sizes, metadata structure, namespace requirements)
   - Common mistakes to avoid (nested metadata, hardcoded keys, missing namespaces)

2. **Consult language-specific guides** for implementation details:
   - **Python**: `.agents/PINECONE-python.md`
   - **TypeScript/Node.js**: `.agents/PINECONE-typescript.md`
   - **Go**: `.agents/PINECONE-go.md`
   - **Java**: `.agents/PINECONE-java.md`
   - **CLI**: `.agents/PINECONE-cli.md`
   - **Quickstart**: `.agents/PINECONE-quickstart.md`
   - **Troubleshooting**: `.agents/PINECONE-troubleshooting.md`

See `AGENTS.md` for the mandatory rules.

## Project Setup

To initialize the Pinecone agent documentation and dependencies:

```bash
./install-agent.sh
```

This downloads the latest Pinecone agents reference implementation and configures `AGENTS.md`.

## Key Constraints & Requirements

When implementing features:

- **Always use namespaces** for data isolation (user IDs, sessions, or content categories)
- **Metadata must be flat** - no nested objects allowed
- **Batch sizes**: Max 96 records for text operations, 1000 for vector operations (also 2MB limit per batch)
- **Metadata per record**: 40KB maximum
- **Consistency model**: Eventually consistent with 1-5 second delay after upsert
- **Environment variables**: Use `.env` files, never hardcode API keys

## Search Quality Best Practices

- Use reranking with `bge-reranker-v2-m3` model for production search
- Request 2x candidates initially, then rerank to get final result count
- This pattern demonstrated in `.agents/PINECONE-quickstart.md`

## Next Steps When Adding Application Code

When building out the recipe search application:

1. **Determine the programming language** (check for package.json, requirements.txt, etc., or ask user)
2. **Choose index creation approach** from `.agents/PINECONE.md#🎯-MANDATORY-Index-Creation-Choose-Your-Approach`
3. **Reference language-specific guide** for SDK installation and implementation patterns
4. **Implement recipe data operations**: upsert, search, and metadata filtering
5. **Add error handling** with exponential backoff for transient failures
6. **Use namespaces** (e.g., per-user recipe collections)
