import pandas as pd
import numpy as np
import pickle
import requests
import time

# ─────────────────────────────────────────────
# Daily Values (adults >= 4 years)
# ─────────────────────────────────────────────
DAILY_VALUES = {
    'protein': 50,
    'fat': 78,
    'saturated_fat': 20,
    'cholesterol': 300,
    'carbohydrates': 275,
    'sodium': 2300,
    'fiber': 28,
    'added_sugars': 50,
    'vitamin_a': 900,
    'vitamin_c': 90,
    'calcium': 1300,
    'iron': 18,
    'vitamin_d': 20,
    'vitamin_e': 15,
    'vitamin_k': 120,
    'potassium': 4700,
    'magnesium': 420,
    'zinc': 11,
}

NUTRIENT_MAP = {
    'Protein': ('protein', 50),
    'Total lipid (fat)': ('fat', 78),
    'Carbohydrate, by difference': ('carbohydrates', 275),
    'Sodium, Na': ('sodium', 2300),
    'Fiber, total dietary': ('fiber', 28),
    'Calcium, Ca': ('calcium', 1300),
    'Iron, Fe': ('iron', 18),
    'Potassium, K': ('potassium', 4700),
    'Vitamin C, total ascorbic acid': ('vitamin_c', 90),
    'Vitamin A, RAE': ('vitamin_a', 900),
    'Cholesterol': ('cholesterol', 300),
    'Fatty acids, total saturated': ('saturated_fat', 20),
    'Magnesium, Mg': ('magnesium', 420),
    'Zinc, Zn': ('zinc', 11),
}

API_KEY = 'ZMYqTcH7YqVNqgcN9YEUtuFmnGxBLWKPS55M9QIw'


class RatingPredictor:
    """Predicts rating class (bad/so-so/great) for a list of ingredients."""

    def __init__(self, model_path='data/best_model.pkl',
                 cols_path='data/ingredient_cols.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(cols_path, 'rb') as f:
            self.ingredient_cols = pickle.load(f)

    def predict(self, ingredients):
        """
        ingredients: list of strings (ingredient names)
        returns: string - 'bad', 'so-so', or 'great'
        """
        # Create feature vector
        features = np.zeros(len(self.ingredient_cols))
        ingredients_lower = [i.strip().lower() for i in ingredients]
        for idx, col in enumerate(self.ingredient_cols):
            if col.lower() in ingredients_lower:
                features[idx] = 1.0
        prediction = self.model.predict([features])[0]
        return prediction


class NutritionFacts:
    """Fetches and displays nutrition facts for ingredients."""

    def __init__(self, nutrition_path='data/nutrition_facts.csv',
                 api_key=API_KEY):
        self.api_key = api_key
        try:
            self.nutrition_df = pd.read_csv(nutrition_path, index_col='ingredient')
        except FileNotFoundError:
            self.nutrition_df = pd.DataFrame()

    def _fetch_from_api(self, ingredient):
        """Fetch nutrition data from USDA API."""
        url = 'https://api.nal.usda.gov/fdc/v1/foods/search'
        params = {
            'query': ingredient,
            'api_key': self.api_key,
            'pageSize': 1
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if not data.get('foods'):
                return {}
            food = data['foods'][0]
            nutrients = {n['nutrientName']: n['value']
                         for n in food.get('foodNutrients', [])}
            return nutrients
        except Exception:
            return {}

    def get_facts(self, ingredient):
        """
        Returns dict of nutrient_name -> % daily value for one ingredient.
        """
        ing_lower = ingredient.strip().lower()

        # Check cached data first
        if ing_lower in self.nutrition_df.index:
            row = self.nutrition_df.loc[ing_lower]
            result = {}
            for col in row.index:
                val = row[col]
                if pd.notna(val) and val > 0:
                    nutrient_name = col.replace('_pct_dv', '').replace('_', ' ').title()
                    result[nutrient_name] = round(float(val), 1)
            return result

        # Fetch from API
        raw = self._fetch_from_api(ing_lower)
        result = {}
        for api_name, (col_name, dv) in NUTRIENT_MAP.items():
            val = raw.get(api_name, 0) or 0
            pct = round(val / dv * 100, 1)
            if pct > 0:
                display_name = col_name.replace('_', ' ').title()
                result[display_name] = pct
        return result

    def display(self, ingredients):
        """
        Print nutrition facts for a list of ingredients.
        """
        for ingredient in ingredients:
            print(ingredient.strip().title())
            facts = self.get_facts(ingredient)
            if facts:
                for nutrient, pct in facts.items():
                    print(f'  {nutrient} - {pct}% of Daily Value')
            else:
                print('  No nutrition data available')


class SimilarRecipes:
    """Finds most similar recipes to a list of ingredients."""

    def __init__(self, recipes_path='data/recipes_with_urls.csv'):
        self.recipes_df = pd.read_csv(recipes_path)
        # Get ingredient columns (binary 0/1)
        self.ingredient_cols = [c for c in self.recipes_df.columns
                                 if c not in ['title', 'rating', 'url']]

    def find_similar(self, ingredients, top_n=3):
        """
        Find top_n most similar recipes based on ingredient overlap.
        Returns list of dicts with title, rating, url.
        """
        ingredients_lower = set(i.strip().lower() for i in ingredients)

        # Calculate Jaccard similarity
        scores = []
        for _, row in self.recipes_df.iterrows():
            recipe_ingredients = set(
                col for col in self.ingredient_cols
                if row[col] == 1.0
            )
            if not recipe_ingredients:
                continue
            intersection = len(ingredients_lower & recipe_ingredients)
            union = len(ingredients_lower | recipe_ingredients)
            similarity = intersection / union if union > 0 else 0
            scores.append((similarity, row['title'], row['rating'], row['url']))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, title, rating, url in scores[:top_n]:
            results.append({
                'title': title,
                'rating': rating,
                'url': url,
                'similarity': sim
            })
        return results

    def display(self, ingredients, top_n=3):
        """Print top-3 similar recipes."""
        similar = self.find_similar(ingredients, top_n)
        for i, recipe in enumerate(similar, 1):
            print(f'- {recipe["title"]}, rating: {recipe["rating"]}, '
                  f'URL: {recipe["url"]}')
