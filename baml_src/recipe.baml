class Recipe {
    id string @description(#"
        The unique identifier of the recipe. This is the slug of the recipe. This should be the same as the _id field in the vector DB schema. This should be a lowercase string with hyphens between words. This should be derived from the title of the recipe, not the contents.
        For example, "caribbean-chicken" for the recipe "Caribbean Chicken".
    "#)
    content string @description(#"
        The full recipe content. This is the full recipe content, including the title, ingredients, instructions, and other information.
    "#)
    title string @description(#"
        The title of the recipe.
    "#)
    ingredients string @description(#"
        The ingredients of the recipe. This is the ingredients section of the recipe. It should be presented as a single string with a numbered list in it. For example:
        ```
        1. 1 cup of flour
        2. 2 cups of sugar
        3. 3 cups of milk
        ```
    "#)
    instructions string @description(#"
        The instructions of the recipe. This is the instructions section of the recipe. It should be presented as a single string with a numbered list in it. For example:
        ```
        1. Preheat the oven to 350 degrees Fahrenheit.
        2. In a large bowl, mix together the flour, sugar, and milk.
        3. Pour the mixture into a 9x5 inch loaf pan.
        4. Bake for 1 hour to 1 hour and 15 minutes, or until a toothpick inserted into the center of the loaf comes out clean.
        ```
    "#)
    rating float | null @description(#"
        The rating of the recipe. This is a float between 1.0 and 10.0. Return null only if you truly cannot evaluate the recipe.

        RATING CRITERIA (rate based on how well the recipe fits these preferences):

        Dietary Preferences:
        - Avoid dairy (cheese, milk, cream) except in: pizza, ice cream, some yogurts. Dairy should be optional/on-the-side, not integral.
        - Avoid eggs as a main component (eggs in baked goods are fine).
        - Moderate spice levels preferred. Should be GERD-friendly: not too spicy, not too fatty, not too citrusy, not too meat-heavy.
        - Limit red meat (hereditary colon cancer susceptibility). Other meats and seafood are good.
        - Seafood/fish: all types are enjoyed.

        Cuisine Preferences (favor these):
        - Favorites: Indian, East Asian (Chinese, Japanese, Korean, Thai), Spanish/Iberian, Cajun
        - Also liked: Mexican, Italian (but dairy limits selection)
        - Least favorite: bland cuisines (English, Ecuadorian)

        Practical Considerations:
        - Favor recipes with 40-45 minutes or less total prep+cook time
        - Favor recipes with 10 or fewer ingredients
        - Favor recipes without expensive or hard-to-find ingredients

        Scoring guidance:
        - 8-10: Excellent fit (preferred cuisine, no dietary issues, quick/easy)
        - 6-7: Good fit (minor issues or neutral cuisine)
        - 4-5: Acceptable but has drawbacks (contains dairy/eggs as optional, longer prep, etc.)
        - 1-3: Poor fit (integral dairy/eggs, very fatty/spicy, bland cuisine, excessive red meat)
    "#)
    diet string[] @description(#"
        The diet of the recipe. This is an array of strings. The possible values are: vegetarian, vegan, pescatarian, dairy-free, gluten-free. You may add new values if the recipe clearly fits a dietary category not listed (e.g., "keto", "paleo").
    "#)
    protein string[] @description(#"
        The protein of the recipe. This is an array of strings. The possible values are: beef, pork, lamb, chicken, turkey, seafood, tofu, egg, none. 
        "seafood" covers fish, shrimp, clams, etc.
        Use "none" for recipes with no significant protein source
        A recipe can have multiple proteins (e.g., jambalaya with ["chicken", "seafood", "pork"])
        You may add new values if needed (e.g., "duck", "game")
    "#)
    cuisine string[] @description(#"
        The cuisine of the recipe. This is an array of strings. The possible values are: american, cajun, caribbean, chinese, cuban, indian, italian, japanese, korean, mexican, thai, african, brazilian, mediterranean, other. Use "other" sparingly - prefer adding a specific cuisine if identifiable. A recipe can blend cuisines (e.g., ["thai", "chinese"] for a fusion dish). You may add new values if needed (e.g., "indonesian", "vietnamese", "spanish"). Only use "other" if you truly cannot identify the national or ethnic origin of the recipe.
    "#)
    meal_type string[] @description(#"
        The meal type of the recipe. This is an array of strings. The possible values are: main-dish, side-dish, dessert, sauce, soup, salad, breakfast, snack. A recipe can be multiple types (e.g., a substantial salad could be ["salad", "main-dish"]). You may add new values if needed (e.g., "appetizer", "beverage", "condiment").
    "#)
    difficulty string @description(#"
        The difficulty of the recipe. The possible values are: easy, medium, hard. Unlike with some of the other categories, this is mutually exclusive.
        easy: Few ingredients, simple techniques, under 30 mins active time
        medium: More ingredients or techniques, 30-60 mins active time
        hard: Complex techniques, many steps, over 60 mins active time, or requires special skills
        You may add new values if needed (e.g., "expert", "advanced")
    "#)
    prepTimeMinutes int @description(#"
        The preparation and cooking time of the recipe in minutes. Estimate based on the instructions. Include both prep time and cooking/baking time. Return null if you cannot reasonably estimate.
    "#)
}

function ExtractRecipe(recipe: string) -> Recipe {
    client "CustomGPT5Mini"
    prompt #"
        Extract the recipe from the following content. Normalize quantities into decimals. For instance, 1 and a half cups flour should be 1.5 cups flour, 1/2 teaspoon salt should be 0.5 teaspoons salt, etc.:
        {{ recipe }}

        {{ ctx.output_format }}
    "#
}

function ExtractRecipeFromImage(image: image) -> Recipe {
    client "CustomGPT4oVision"
    prompt #"
        {{ _.role("user") }}
        Extract the recipe from the following image. Normalize quantities into decimals. For instance, 1 and a half cups flour should be 1.5 cups flour, 1/2 teaspoon salt should be 0.5 teaspoons salt, etc.:
        {{ image }}

        {{ ctx.output_format }}
    "#
}

function ExtractRecipeFromImages(images: image[]) -> Recipe {
    client "CustomGPT4oVision"
    prompt #"
        {{ _.role("user") }}
        Extract the recipe from the following images. These images represent pages of a multi-page document (such as a PDF).
        Combine information from all pages to extract the complete recipe.
        Normalize quantities into decimals. For instance, 1 and a half cups flour should be 1.5 cups flour, 1/2 teaspoon salt should be 0.5 teaspoons salt, etc.:

        {% for img in images %}
        Page {{ loop.index }}:
        {{ img }}
        {% endfor %}

        {{ ctx.output_format }}
    "#
}

test TestExtractRecipe {
  functions [ExtractRecipe]
  args {
    recipe #"
      Caribbean Chicken

    1 teaspoon paprika  
    1 teaspoon onion powder  
    1 teaspoon garlic powder  
    1 teaspoon dried parsley  
    1/2 teaspoon dried oregano  
    1 teaspoon salt  
    1 teaspoon pepper  
    4 boneless, skinless chicken breast halves  
    1/4 cup duck sauce  
    1/4 cup marinara sauce  
    1 teaspoon mango hot sauce  
    3/4 cup fresh pink grapefruit juice, divided  
    1 cup Italian seasoned bread crumbs  
    1 ripe nectarine, pitted and sliced

    Preheat oven to 375 degrees F (190 degrees C). Line a baking dish with parchment paper.

    In a large bowl, mix together paprika, onion powder, garlic powder, parsley, oregano, salt and pepper. Toss with chicken breasts until evenly coated. In a bowl, combine duck sauce, marinara sauce, mango hot sauce and 1/4 cup grapefruit juice. Coat chicken evenly with sauce mixture. Place bread crumbs in a shallow dish, and dredge chicken until evenly breaded. Place chicken in baking dish. Place nectarine slices around the edge of baking dish.

    Bake in preheated oven for 15 minutes. Turn the chicken, pour 1/2 cup grapefruit juice over chicken, and continue cooking for another 15 minutes or until done.
    "#
  }
}


test TestExtractRecipeFromImage {
  functions [ExtractRecipeFromImage]
  args {
    image {
      file "./IMG_1961.jpg"
    }
  }
}