# Add Sparse Search Capabilities

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents

Right now, the application just performs semantic searches when you Run a search. "For example, give me a good seafood recipe for a weeknight". Enabling dense searches might enable more hits. This might help with the problem where searches that are too complex don't return high enough values from the vector database. For example, a search like "Give me a good weeknight recipe that two people can cook together in less than half an hour, with meat in it" Currently does not hit the vector database. But I do think this applies to some of my recipes. A sparse search might help with this.  

## Acceptance criteria

- A sparse search capability will be implemented in addition to the existing semantic search capability. 
- There will be logic dictating when to use dense search and when to use semantic search. As a general rule, we might use semantic search first, but if there aren't enough hits, we might fall back to a sparse search. This wanted to be worked out. 
- More complex queries will yield a greater number of hits from the vector database. 
