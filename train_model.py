"""
Model Training Script

Dataset Acknowledgement:
The dataset used for this project is the 'Food Ingredients and Recipe Dataset with Image Name Mapping,'
originally available on Kaggle through Sakshi Goel. The original author is unknown.

This script performs the following tasks:
1. Loads the dataset.
2. Vectorizes the 'Cleaned_Ingredients' column using TF-IDF.
3. Trains a KMeans clustering model.
4. Saves the trained models and vectorizer.
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib


# Ensure Model directory exists
if not os.path.exists('Model'):
    os.makedirs('Model')

# Load the dataset
df = pd.read_csv('Food Ingredients and Recipe Dataset with Image Name Mapping.csv')

# Ensure the dataset contains the required columns
required_columns = ['Title', 'Ingredients', 'Instructions', 'Image_Name', 'Cleaned_Ingredients']
for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Dataset must contain the '{column}' column")

# Vectorize the Cleaned_Ingredients column
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Cleaned_Ingredients'])

# Train the KMeans model
num_clusters = 10
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_model.fit(tfidf_matrix)

# Save the trained models
joblib.dump(vectorizer, 'Model/tfidf_vectorizer.pkl')
joblib.dump(tfidf_matrix, 'Model/tfidf_matrix.pkl')
joblib.dump(kmeans_model, 'Model/kmeans_model.pkl')

print("Model training complete and models saved.")
