from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from flask_cors import CORS

# Load dataset
movies = pd.read_csv("processed_movies.csv")

# ✅ TF-IDF Vectorizer & Cosine Similarity
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['combined_features'])
tfidf_cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ✅ Train Word2Vec Model (on movie descriptions)
tokenized_descriptions = movies['combined_features'].dropna().apply(lambda x: x.split())
word2vec_model = Word2Vec(sentences=tokenized_descriptions, vector_size=100, window=5, min_count=2, workers=4)

def get_word2vec_vector(text):
    words = text.split()
    vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

# ✅ Compute Word2Vec Cosine Similarity
word2vec_vectors = np.array([get_word2vec_vector(desc) for desc in movies['combined_features'].fillna('')])
word2vec_cosine_sim = cosine_similarity(word2vec_vectors, word2vec_vectors)

# ✅ Hybrid Similarity Calculation
def get_hybrid_similarity(tfidf_score, w2v_score, alpha=0.6):
    return alpha * tfidf_score + (1 - alpha) * w2v_score

# Flask App
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for frontend API calls

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["GET"])
def recommend():
    movie_title = request.args.get("movie", "").strip().lower()
    
    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400

    matched_movies = movies[movies['title'].str.lower().str.contains(movie_title, regex=False)]
    
    if matched_movies.empty:
        return jsonify({"error": "Movie Not Found"}), 404

    movie_index = matched_movies.index[0]
    
    # ✅ Calculate Hybrid Similarity Scores
    tfidf_scores = tfidf_cosine_sim[movie_index]
    w2v_scores = word2vec_cosine_sim[movie_index]
    hybrid_scores = get_hybrid_similarity(tfidf_scores, w2v_scores)
    
    sorted_movies = sorted(enumerate(hybrid_scores), key=lambda x: x[1], reverse=True)[1:6]

    # 🎬 Get details of input movie
    input_movie = movies.iloc[movie_index]
    input_movie_details = {
        "title": input_movie["title"],
        "overview": input_movie.get("overview", "No overview available"),
        "rating": float(input_movie.get("vote_average", 0)),
        "budget": int(input_movie.get("budget", 0)),
        "revenue": int(input_movie.get("revenue", 0)),
        "cast": input_movie.get("cast", "").split(",")[:4] if isinstance(input_movie.get("cast"), str) else [],
    }

    # 🎬 Get recommended movies
    recommendations = []
    for i in sorted_movies:
        movie_data = movies.iloc[i[0]]
        recommendations.append({
            "title": movie_data["title"],
            "overview": movie_data.get("overview", "No overview available"),
            "rating": float(movie_data.get("vote_average", 0)),
            "budget": int(movie_data.get("budget", 0)),
            "revenue": int(movie_data.get("revenue", 0)),
            "cast": movie_data.get("cast", "").split(",")[:4] if isinstance(movie_data.get("cast"), str) else [],
        })

    return jsonify({
        "movie": input_movie_details,
        "recommendations": recommendations
    })

if __name__ == "__main__":
    app.run(debug=True)
