"""
LLM helper utilities for recipe chatbot.

Provides functions for:
- Generating recipe responses from RAG results
- Generating fallback recipes when no match is found
"""

from openai import OpenAI


def generate_recipe_response(
    user_query: str,
    rag_results: list[dict],
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a response to the user's query based on RAG search results.

    Uses the LLM to create a natural language response that presents
    the best matching recipe(s) from the user's collection.

    Args:
        user_query: The user's original question
        rag_results: List of recipe hits from Pinecone search
        client: OpenAI client instance
        model: Chat completion model to use

    Returns:
        Natural language response presenting the recipe(s)
    """
    if not rag_results:
        return "No recipes found in your collection matching your query."

    # Format the top results for the prompt
    recipes_context = _format_recipes_for_prompt(rag_results)

    system_prompt = """You are a helpful recipe assistant. The user has a personal recipe collection,
and you help them find recipes that match their requests.

When presenting a recipe:
1. Start with a brief introduction explaining why this recipe matches their request
2. Present the full recipe with ingredients and instructions
3. If the recipe has a rating, mention it
4. Keep your response focused and practical

Do not add recipes or ingredients that aren't in the provided data."""

    user_prompt = f"""The user asked: "{user_query}"

Here are the matching recipes from their collection:

{recipes_context}

Please present the best matching recipe to the user in a helpful way."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=1500,
    )

    return response.choices[0].message.content


def generate_fallback_recipe(
    user_query: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a recipe when no suitable match is found in the RAG database.

    Uses the LLM to create a new recipe based on the user's request.

    Args:
        user_query: The user's original question
        client: OpenAI client instance
        model: Chat completion model to use

    Returns:
        Generated recipe response
    """
    system_prompt = """You are a helpful recipe assistant. When the user's personal recipe collection
doesn't have a suitable match, you generate a new recipe for them.

When generating a recipe:
1. Create a practical, easy-to-follow recipe that matches the user's request
2. Use common ingredients that are easy to find
3. Format with clear sections: Title, Ingredients (bulleted), Instructions (numbered)
4. Include approximate prep/cook time
5. Keep it realistic - don't suggest overly complex techniques

Be creative but practical."""

    user_prompt = f"""The user asked: "{user_query}"

Unfortunately, their personal recipe collection doesn't have a suitable match.
Please generate a recipe that matches their request."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        max_tokens=1500,
    )

    return response.choices[0].message.content


def _format_recipes_for_prompt(rag_results: list[dict], max_recipes: int = 3) -> str:
    """
    Format RAG results into a string for the LLM prompt.

    Args:
        rag_results: List of recipe hits from Pinecone search
        max_recipes: Maximum number of recipes to include

    Returns:
        Formatted string with recipe details
    """
    formatted = []

    for i, hit in enumerate(rag_results[:max_recipes], 1):
        fields = hit.get("fields", {})
        score = hit.get("_score", 0)

        recipe_str = f"--- Recipe {i} (relevance score: {score:.2f}) ---\n"

        # Get title from metadata or fall back to _id
        title = fields.get("title") or fields.get("metadata", {}).get("title") or hit.get("_id", "Unknown")
        recipe_str += f"Title: {title}\n"

        # Get rating
        rating = fields.get("rating") or fields.get("metadata", {}).get("rating")
        if rating:
            recipe_str += f"Rating: {rating}/10\n"

        # Get content (full recipe text)
        content = fields.get("content", "")
        if content:
            recipe_str += f"\n{content}\n"

        formatted.append(recipe_str)

    return "\n".join(formatted)


def check_score_threshold(
    rag_results: list[dict],
    threshold: float = 0.10,
) -> tuple[bool, float, list[dict]]:
    """
    Check if the top RAG result meets the score threshold.

    Explicitly sorts results by score to find the highest, rather than
    assuming the API returns them in sorted order. Future iterations
    may add more sophisticated evaluation logic (e.g., LLM verification).

    Args:
        rag_results: List of recipe hits from Pinecone search
        threshold: Minimum score to consider a match

    Returns:
        Tuple of (passes_threshold, top_score, sorted_results)
    """
    if not rag_results:
        return False, 0.0, []

    sorted_results = sorted(
        rag_results,
        key=lambda x: x.get("_score", 0),
        reverse=True,
    )
    top_score = sorted_results[0].get("_score", 0)
    return top_score >= threshold, top_score, sorted_results
