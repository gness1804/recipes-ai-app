#!/bin/bash
set -e

# Pin to a specific release tag rather than "latest" to avoid supply-chain risk.
# Update RELEASE_TAG and EXPECTED_SHA256 together when upgrading.
RELEASE_TAG="v0.0.1"
EXPECTED_SHA256=""  # Set after first verified download; leave empty to skip check.

DOWNLOAD_URL="https://github.com/pinecone-io/pinecone-agents-ref/releases/download/${RELEASE_TAG}/agents.zip"

curl -L -o agents.zip "$DOWNLOAD_URL"

if [ -n "$EXPECTED_SHA256" ]; then
    ACTUAL_SHA256=$(shasum -a 256 agents.zip | awk '{print $1}')
    if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
        echo "ERROR: SHA-256 mismatch!"
        echo "  Expected: $EXPECTED_SHA256"
        echo "  Actual:   $ACTUAL_SHA256"
        rm agents.zip
        exit 1
    fi
    echo "SHA-256 verified."
fi

unzip agents.zip && rm agents.zip
touch AGENTS.md && cat AGENTS-pinecone-snippet.md >> AGENTS.md && rm AGENTS-pinecone-snippet.md
