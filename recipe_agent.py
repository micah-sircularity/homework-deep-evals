from pydantic_ai import Agent
from model_config import MODEL
from recipe_tools import search_recipes, get_recipe

instructions = """You are a recipe assistant. You help users find recipes and answer cooking questions.
1. Use search_recipes to find recipes matching the user's request
2. Use get_recipe to get full details including instructions
3. Answer based on the recipe data you have - do not make up recipes or ingredients
4. If asked about something not in the recipe collection, say you don't have that recipe
5. You can suggest alternatives from the collection if you don't have an exact match
"""

agent = Agent(
    MODEL,
    tools=[search_recipes, get_recipe],
    instructions=instructions,
)
