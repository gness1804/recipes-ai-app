---
github_issue: 13
---
# Recipe match threshold is too low.

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents

The threshold for matching recipes in this application in the RAG vector database is too low. As you can see from the screenshots, I am performing searches that should turn up recipes from my database, but the threshold matching is very low, so it's not returning any recipes from the database. I think a search such as "a good seafood recipe" or "a good vegetarian recipe that doesn't take long to cook" should have a number of good candidates in my recipes. So I think the threshold is too low, and these recipes aren't being found. 

We may need to make changes such as adjusting the threshold, adding more recipes, or something else. There might also be some configuration we can do within Pinecone. You will need to work with me on this to ensure that we get better matches. 

<img width="1220" height="677" alt="Image" src="https://github.com/user-attachments/assets/2241a694-d432-4fb1-a5e6-14364ba2b49e" />

<img width="610" height="337" alt="Image" src="https://github.com/user-attachments/assets/f4b11063-365c-430d-ae26-6a458943114e" />



## Acceptance criteria

<!-- DONE -->
