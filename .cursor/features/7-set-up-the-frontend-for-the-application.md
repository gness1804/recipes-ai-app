---
github_issue: 11
---
# Set Up The Frontend For The Application

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents

We need to set up the frontend for this application after the initial MVP is done. This will consist of a modern and attractive user interface in which the user enters a question related to finding a recipe. Examples might include:
- Fetch me a good weeknight recipe that's vegetarian
- I want a nice seafood recipe to make for a date night

The back-end logic will then work and emit a resulting recipe, which the user will see. 
There should also be a way to save prior queries and their results initially in local storage. And also a button that you can press that erases this history. 

## Acceptance criteria

- A modern user interface for asking a query for a recipe and then getting a recipe back. 
- The ability to save prior chats, list them, load them, and delete all prior chats. 
- Error handling including cases such as:
	- The user asking a question irrelevant to recipes
	- Standard problems such as a back-end or LLM failure
