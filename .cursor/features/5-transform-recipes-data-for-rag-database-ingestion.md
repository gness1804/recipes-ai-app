---
github_issue: 4
---
# Transform Recipes Data For RAG Vector Database Ingestion

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents

We need to take the existing recipe data in data/processed-recipes and transform it into a format that can be ingested into a vector database (Pinecone). The target schema that I would like to use is in data/schemas/basicSchema.json. I need to be able to convert the processed recipes from that directory into a format that matches the basic schema. 

A Python script should be able to extract all the information from the recipes for some of these fields. However, other fields might require an LLM to determine, for instance, the diet and protein fields of the recipe. I would rather not add more manual metadata like I did for the rating for those things. So an LLM will probably have to read each recipe and determine the values of these fields for each. 

Present a series of steps to complete this task. First, I would like you to look over my basic schema and let me know if you think it's ideal or optimal for ingestion into a RAG vector database. And if not, what should this schema look like instead? It's important to get the schema right to maximize the effectiveness of our database and app. 

## Acceptance criteria
- A Python script and/or other mechanism to convert the recipes to fit a schema that will be optimized for RAG vector database ingestion. 
- The recipes should be transformed into a format that matches the basic schema or a modified schema that is optimized for RAG vector database ingestion.
