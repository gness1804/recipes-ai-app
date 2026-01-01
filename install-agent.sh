#!/bin/bash
set -e
curl -L -o agents.zip https://github.com/pinecone-io/pinecone-agents-ref/releases/latest/download/agents.zip
unzip agents.zip && rm agents.zip
touch AGENTS.md && cat AGENTS-pinecone-snippet.md >> AGENTS.md && rm AGENTS-pinecone-snippet.md
