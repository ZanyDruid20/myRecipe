import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
import requests
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)
APP_ID = 'da0f4e1f'
APP_KEY = '6ebdb81c8ce9771bf7487d4a8f91bcb2'

# Load the machine learning models
vectorizer = joblib.load('Model/tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('Model/tfidf_matrix.pkl')
kmeans_model = joblib.load('Model/kmeans_model.pkl')

# Load the dataset
recipes_df = pd.read_csv('Food Ingredients and Recipe Dataset with Image Name Mapping.csv')


def fetch_requests_api(query):
    url = f'https://api.edamam.com/search?q={query}&app_id={APP_ID}&app_key={APP_KEY}'
    response = requests.get(url)
    data = response.json()
    return data['hits']


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    recipes = fetch_requests_api(query)
    return jsonify(recipes)


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    userIngredients = data.get('Ingredients')
    if not userIngredients:
        return jsonify({'error': 'Ingredients are required'}), 400

    # Vectorize the Ingredients
    user_vector = vectorizer.transform([userIngredients])

    # Predict the model
    cluster = kmeans_model.predict(user_vector)[0]

    # Find similar recipes within the same cluster
    similar_recipes_indices = [i for i, label in enumerate(kmeans_model.labels_) if label == cluster]

    # Compute similarities
    similarities = cosine_similarity(user_vector, tfidf_matrix[similar_recipes_indices])
    similar_recipes_indices = [similar_recipes_indices[i] for i in similarities.argsort()[0][-10:]]  # Top 10

    # Example result structure
    results = []
    for i in similar_recipes_indices:
        recipe = recipes_df.iloc[i]
        try:
            cleaned_ingredients = json.loads(recipe['Cleaned_Ingredients'])
        except json.JSONDecodeError:
            cleaned_ingredients = recipe['Cleaned_Ingredients'].split(',')  # Fallback

        results.append({
            'Title': recipe['Title'],
            'Cleaned_Ingredients': cleaned_ingredients
        })

    return jsonify({"recommended_recipes": results})


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.route("/")
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
