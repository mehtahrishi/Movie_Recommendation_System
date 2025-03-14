import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load processed dataset
movies = pd.read_csv("processed_movies.csv")

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['combined_features'])

# Compute similarity scores
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to find movies containing a keyword (e.g., "Captain America")
def find_movies_by_keyword(keyword):
    return movies[movies['title'].str.contains(keyword, case=False, na=False)]['title'].tolist()

# Function to get recommendations
def recommend(movie_title):
    # Find all movies related to the keyword
    matched_movies = find_movies_by_keyword(movie_title)

    if not matched_movies:
        print(f"Movie '{movie_title}' not found in dataset.")
        return []

    recommendations = set(matched_movies)  # Store results in a set to avoid duplicates

    for matched_movie in matched_movies:
        movie_index = movies[movies['title'] == matched_movie].index[0]
        similarity_scores = list(enumerate(cosine_sim[movie_index]))
        sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]  # Get top 5
        
        for i in sorted_movies:
            recommendations.add(movies.iloc[i[0]]['title'])  # Add similar movies

    return list(recommendations)

# Test the recommendation function
user_movie = "Captain America"
print(f"Movies similar to '{user_movie}': {recommend(user_movie)}")
print(recommend("The Dark Knight"))
print(recommend("Inception"))
print(recommend("Titanic"))
print(recommend("Endgame"))