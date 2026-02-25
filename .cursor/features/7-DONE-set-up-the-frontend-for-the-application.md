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

We should probably build this in streamlit OR in fast API. (For models, look at Receipt Ranger and Friendly Advice Columnist respectively.) We need to decide which deployment model is more appropriate for this particular project. And then deployment needs to be as simple and inexpensive as possible. 
 
We should also use a bring your own API key model modeled on one of those other projects. There should be something like a secure cookie and an encrypted cookie used to save the API key for a period of seven days. So that users aren't constantly having to re-enter their API key.

The front end should be built out with an eye to eventually accommodating the advanced features described in .cursor/progress/9 (CFS issue 9). 

## Acceptance criteria

- A modern user interface for asking a query for a recipe and then getting a recipe back. 
- The ability to save prior chats, list them, load them, and delete all prior chats. 
- Error handling including cases such as:
	- The user asking a question irrelevant to recipes
	- Standard problems such as a back-end or LLM failure
- Deployment will happen in an inexpensive and easy-to-use platform.

<!-- DONE -->
