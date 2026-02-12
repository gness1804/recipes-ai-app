# Fine Tune To Users Tastes By Asking Them To Review Online Recipes

## Working directory

`~/Desktop/pinecone/recipes-ai-app`

## Contents

This will be a big feature that will only be possible once the site is deployed. For users using an API key that's not mine, it would be useful to be able to show them several recipes and have them rate the recipes. This information would then be fed into the LLM to inform the LLM of the user's taste. This would allow the LLM to suggest more appropriate and relevant recipes in the app. I envision this functionality as existing for users who are not me. When I use my own API key, the application should behave as it does now, where it looks into the RAG database to find a matching recipe, and then generates one only if it can't find a suitable recipe in the RAG database. I have logic depending on different API keys in my project ~/Desktop/receipt-ranger.

There will be several problems that would have to be solved in implementing this feature:
- How to persist the data on users' tastes. Without user accounts, this will be tricky. We might have a big JSON file that maps users' tastes based on their responses and uses a hash ID for each user. But if the application scales, we really need to have some sort of database. This introduces other questions of cost and maintenance.  
- How to test this feature since I won't have access to other peoples' API keys. I might resolve this by using a feature flag, such as OTHER_USER_TESTING_MODE. If turned on, this feature flag would make the application behave as though I were another user, that is, a user without my API key. 
- How to procure the recipes? One option would simply be for the application to look online for a random recipe and then have the users rate it. Another possibility is for the application to store a collection of recipes and ask users to rate these. This might be more useful because you could train the application on what each recipe means for users' tastes.   

## Acceptance criteria

- For users using an API key that's not mine, the application will show them several recipes and ask them to rate these recipes. 
- The application will then take this rating and use it to determine the user's tastes. This, in turn, will inform what recipes the application spits out to the user in response to queries. For example, if Johnny gives a very high rating to meat recipes and a low rating to vegetarian recipes, we might want to show him recipes involving meat. 
- The application will persist each user's recipe taste details somehow, such as in a JSON file, or perhaps later on, in a database. 
- When I am using my own API key, the application will behave as it does today, consulting the RAG database to try to find a matching recipe that matches my query, and only generating a recipe if a suitable match is not found. 
- I have the ability to override this behavior with my own API key and simulate another user by having a testing mode that I can implement probably via a feature flag. 
- I will train the AI to do the work of taking users' responses to recipes and adding that data somewhere to use in response to these users' queries. 
