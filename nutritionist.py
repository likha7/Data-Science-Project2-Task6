#!/usr/bin/env python3
"""
nutritionist.py - Food and Nutrition predictor
Usage: ./nutritionist.py ingredient1, ingredient2, ingredient3
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recipes import RatingPredictor, NutritionFacts, SimilarRecipes


def main():
    if len(sys.argv) < 2:
        print('Usage: ./nutritionist.py ingredient1, ingredient2, ...')
        sys.exit(1)

    # Parse ingredients from command line
    # Handle both: "milk, honey, jam" as one arg or milk honey jam as separate
    raw_input = ' '.join(sys.argv[1:])
    ingredients = [i.strip() for i in raw_input.split(',') if i.strip()]

    if not ingredients:
        print('Please provide at least one ingredient.')
        sys.exit(1)

    # ─────────────────────────────────
    # I. FORECAST
    # ─────────────────────────────────
    print('\nI. OUR FORECAST')
    try:
        predictor = RatingPredictor(
            model_path='data/best_model.pkl',
            cols_path='data/ingredient_cols.pkl'
        )
        rating_class = predictor.predict(ingredients)
        if rating_class == 'great':
            print(f'You might find it tasty! In our opinion, it is a great idea '
                  f'to have a dish with that list of ingredients.')
        elif rating_class == 'so-so':
            print(f'You might find it okay. In our opinion, it is a so-so idea '
                  f'to have a dish with that list of ingredients.')
        else:
            print(f'You might find it tasty, but in our opinion, it is a bad idea '
                  f'to have a dish with that list of ingredients.')
    except Exception as e:
        print(f'Could not make prediction: {e}')
        print('(Run recipes.ipynb first to train and save the model)')

    # ─────────────────────────────────
    # II. NUTRITION FACTS
    # ─────────────────────────────────
    print('\nII. NUTRITION FACTS')
    try:
        nutrition = NutritionFacts(
            nutrition_path='data/nutrition_facts.csv'
        )
        nutrition.display(ingredients)
    except Exception as e:
        print(f'Could not fetch nutrition facts: {e}')

    # ─────────────────────────────────
    # III. TOP-3 SIMILAR RECIPES
    # ─────────────────────────────────
    print('\nIII. TOP-3 SIMILAR RECIPES:')
    try:
        similar = SimilarRecipes(recipes_path='data/recipes_with_urls.csv')
        similar.display(ingredients, top_n=3)
    except Exception as e:
        print(f'Could not find similar recipes: {e}')
        print('(Run recipes.ipynb first to generate recipes_with_urls.csv)')


if __name__ == '__main__':
    main()
