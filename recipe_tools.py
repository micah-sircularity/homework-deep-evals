import json

from minsearch import Index

with open('recipes.json') as f:
    RECIPES = json.load(f)

index = Index(
    text_fields=["name", "cuisine", "instructions", "tags_text", "ingredients_text"],
    keyword_fields=["id", "difficulty", "cuisine"]
)

documents = []

for r in RECIPES:
    doc = dict(r)
    doc['tags_text'] = ', '.join(r['tags'])
    doc['ingredients_text'] = ', '.join(r['ingredients'])
    documents.append(doc)

index.fit(documents)


def search_recipes(query: str, cuisine: str = None) -> str:
    """Search for recipes by name, ingredient, cuisine, or description.

    Args:
        query: What to search for (e.g. 'chicken', 'easy dessert', 'Italian')
        cuisine: Optional filter by cuisine (e.g. 'Italian', 'Thai', 'Mexican')
    """
    filter_dict = {}
    if cuisine:
        filter_dict['cuisine'] = cuisine

    results = index.search(query, filter_dict=filter_dict, num_results=3)

    if not results:
        return "No recipes found matching your search."

    lines = []

    for r in results:
        lines.append(f"[{r['id']}] {r['name']} ({r['cuisine']}, {r['difficulty']})")
        lines.append(f"  Prep: {r['prep_time']}min, Cook: {r['cook_time']}min, Serves: {r['servings']}")
        lines.append(f"  Ingredients: {r['ingredients_text']}")
        lines.append(f"  Tags: {r['tags_text']}")
        lines.append("")

    return "\n".join(lines)


def get_recipe(recipe_id: int) -> str:
    """Get full recipe details including instructions.

    Args:
        recipe_id: The recipe ID number
    """
    for r in RECIPES:
        if r['id'] == recipe_id:
            ingredients = '\n'.join('- ' + i for i in r['ingredients'])
            return f"""Recipe: {r['name']}
Cuisine: {r['cuisine']}
Difficulty: {r['difficulty']}
Prep time: {r['prep_time']} minutes
Cook time: {r['cook_time']} minutes
Servings: {r['servings']}

Ingredients:
{ingredients}

Instructions:
{r['instructions']}

Tags: {', '.join(r['tags'])}"""

    return f"Recipe with ID {recipe_id} not found."
