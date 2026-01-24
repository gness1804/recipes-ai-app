---
github_issue: 8
---
# Make Rating Determined By Llm Rather Than Manually Entered

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents

Right now, the recipe ratings are manually entered, and then applied to each recipe. See data/processed-recipes/_ratings.json. But for version 2, I want an LLM to determine the ratings. Version 2 is in the v2 directory. The LLM should use the following criteria when determining ratings for each recipe: v2/dietary_preferences.md.

This will be part of the revamped process for processing recipes in version 2. This version will use BAML to process recipes instead of what we were doing before, which was a multi-step process. As part of this recipe processing, the LLM that's reading the raw recipe and returning structured data will also perform ratings analysis and deliver a rating.

A big concern with this is cost and token usage. I want to find a way to most effectively use tokens in making this adjustment. So that ideally, for every recipe that I might add in the future, it would not repeat a lot of tokens. Work with me to determine the best way to do this.

This new rating evaluation should only apply to new recipes in v2. All the previously processed recipes in the data/processedrecipes directory should remain as they are. I don't want to change any of those ratings. Bur moving forward, I want to use the LLM system to rate the different recipes that we add to V2. 

Please present a series of steps for me to look over before performing any actual work on it.

## Acceptance criteria

- As part of the process of recipe Processing/transformation in v2, the LLM doing the processing will offer a rating to each recipe based on the rating criteria document outlined above.
- This rating will be returned as part of the structured data that the LLM will return for each recipe.
- This process will minimize token usage and cost.

<!-- DONE -->
