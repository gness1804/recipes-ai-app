# Transform Recipe Data To Optimize It To Be Ingested Into A Rag Database

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents
This project is meant to be an AI chat interface that draws upon a Pinecone rag database. The app will be  a chatbot that the user can chat with about recipes. They can ask questions such as: "Give me a good seafood recipe for a weeknight" or "Give me an easy 30-minute beef recipe with minimal dairy." The app will output a recipe for the user that matches the request. 

The app will draw upon a RAG vector database by Pinecone. The database will contain data on my own personal recipes. This will be the first source of truth of the application. The application will first look at these recipes, evaluate whether any of them are appropriate for the user's question, and if so, choose the best one. If not, the application will either generate a recipe on its own or search the web for one. 

The current task is to transform raw recipes from my Google Drive into recipes that can be ingested by the RAG database. The target document format is reflected in this document: `data/target_document_format.md.` The transformer will take the raw recipes and transform them into this format. The transformer will either be a Python script or an LLM. We need to determine which one is most appropriate. I'm thinking maybe an LLM, because it'll need to take disparately formatted recipes and push them into the same format. But I'm open to what your opinion is. 

The raw recipes that I have so far are in the following directory: `data/raw_recipes`. This will give you an idea about what the raw data looks like. 

## Acceptance criteria
- The transformer will take the raw recipes and transform them into the target document format.
- The transformer will be either a Python script or an LLM.
- The transformer will be determined based on the task requirements and the available resources.
