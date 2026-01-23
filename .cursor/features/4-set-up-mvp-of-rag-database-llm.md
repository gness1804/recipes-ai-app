---
github_issue: 3
---
# Set Up Mvp Of Rag Database Llm

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents

Right now, we have a list of recipes which has been normalized and will go into a RAG vector database. The next step is to actually create an MVP, which will be an application that takes in a recipe-related question. For example, "Give me a good seafood recipe for a weeknight." The application should then use an LLM and spit out an answer. The answer will be based in part on the data in a RAG vector database that I will set up, which will be based on these recipes. The application should first look in the RAG database for a suitable recipe, and if one can't be found, it should then either create one or find one on the web. 

## Acceptance criteria

- The application will take in a user question related to an LLM. (This first phase is just backend. We will add the frontend later.)
- The application will use an LLM to process the user question and emit an appropriate response. 
- The response will be based on a recipe from a group of recipes stored in a vector database, if appropriate. 
- The application will first look in the RAG vector database for a suitable recipe. 
- If none can be found, the application will then look either on the web or create a suitable recipe. 
- Right now, the application doesn't know the user's tastes other than the prompt and the vector database. Later, we will train a model or fine-tune a model on particular recipes and ratings, but right now, for this MVP, I just want it to be a simple base LLM connected to a rag database. 
