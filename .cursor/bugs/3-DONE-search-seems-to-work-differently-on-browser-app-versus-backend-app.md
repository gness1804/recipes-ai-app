---
github_issue: 26
---
# Search seems to work differently on browser app versus backend app. 

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents

When searching for recipes in the browser application versus the backend application, I get quite different results. The two main differences are:
1. The browser app seems fixated on lemon garlic recipes. Almost all the recipes that I've searched for using the browser app have returned lemon garlic something. By contrast, in the backend application, this isn't the case.
2. The browser app seems to always use a dense search, while the backend app seems to mostly use sparse search. 

This might be based on the parameters. Are the parameters different for the deployed application versus the backend application?


## Acceptance criteria


- Both the browser app and the backend search will return similar results.

<!-- DONE -->
