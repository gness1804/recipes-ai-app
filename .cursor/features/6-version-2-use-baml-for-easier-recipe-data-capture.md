# Version 2 Use Baml For Easier Recipe Data Capture

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents

Right now, there's a convoluted process for taking a raw recipe, whether in a doc format, Markdown, image, or PDF, and transforming it into a particular schema. I just learned about BAML, Which is a tool to make LLM more deterministic. I believe that I can use this to eliminate most of the middle processes and simply take a raw recipe and transform it directly into the target schema. The schema lives at data/schemas/basicSchema.json. I think I can use BAML to transform the recipe data into the schema by creating a class that has the properties of the schema, and then simply running a BAML function against a given recipe. We might need to use a simplified version of this schema as the BAML class. But it still seems a lot less computationally intensive than what we have now. 

## Acceptance criteria

- Application will be able to take a raw recipe in any form and using BAML implementation transform it and spit out strictly structured data on the recipe such as title, ingredients, steps, etc. 
- Translation will have a high degree of accuracy and will be the same every time, deterministic, thanks to BAML. 
- We might need a minimal function to translate a BAML class into a larger schema. 
