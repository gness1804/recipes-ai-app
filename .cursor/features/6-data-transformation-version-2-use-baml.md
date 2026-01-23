# Data Transformation Version 2 Use Baml

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents

Right now, our data transformation process from raw recipe to processed recipe is convoluted. It's a multi-step process that goes from raw data to standardized Markdown to a JSON-like schema. Using BAML should make the process easier. The file baml_src/recipe.baml contains the BAML logic of a recipe schema and a way to take a raw recipe and extract standardized data from it. What we need to do is create a script that uses this one on recipes that will be stored in v2/raw-recipes. These will be photos or text documents of recipes as before. The new script will use BAML to transform these recipes into an example as shown in the following file: v2/CaribbeanChickenExampleBAMLOutput.json.

We will probably then need the same script for our second script to take these JSON examples and transform them into the full JSON schema as shown here: data/schemas/basicSchema.json (unless we can use BAML to translate the raw recipes into that format directly.) The final version of the recipes for the RAG vector database to ingest should be stored in a single file in the v2 directory.

This processing should be separate from the processing that we did for v1. (See the data directory) that is, all those recipes should be considered finished. The v2 will only process new, separate recipes. 

The only big snag I can see is ratings. Most of these recipes won't come with ratings. We might need some similar logic to pinpoint which recipes for v2 don't have ratings so that I can then go and manually add them as before. But every other value for these recipes should be added by the LLM via BAML.

## Acceptance criteria
- Script that takes raw recipes for version 2 and transforms them into a format matching basicSchema using BAML.
- A mechanism that handles ratings, including flagging recipes without ratings and making sure that I rate them if I can.
