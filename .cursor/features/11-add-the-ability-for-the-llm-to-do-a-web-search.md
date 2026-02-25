---
github_issue: 22
---
# Add the ability for the LLM to do a web search. 

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents

Right now, if the LLM is not in owner mode, or if it can't find a recipe in the RAG database, it falls back to generating a recipe only. I would also like there to be an option for the LLM to search the web for recipes. We need to figure out what logic makes sense for web search versus generating search. One option could be to randomize this so that at any given time, it's random whether the LLM does a web search versus a generated search. Also, another option is that the LLM could do a web search, and if nothing sufficient is found, it could just generate something. Even do a hybrid where it can find something on the web, but then modify it according to a user's tastes. There are a number of options here, and it will be worth exploring what makes the most sense for this application. 


## Acceptance criteria
