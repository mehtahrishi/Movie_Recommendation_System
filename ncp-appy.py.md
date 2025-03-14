import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ncf_model import NCF, get_user_recommendations  # Import NCF functions

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend API calls

# Load the trained NCF model
num_users = 4  # Change this to match trained model
num_movies = 5  # Change this to match trained model
  # Adjust as per dataset
model = NCF(num_users, num_movies)
model.load_state_dict(torch.load("ncf_model.pth"))
model.eval()

# Sample movie mapping (Replace with actual database or CSV lookup)
movie_mapping = {
    101: "Inception",
    102: "The Dark Knight",
    103: "Interstellar",
    104: "Fight Club",
    105: "The Matrix"
}

@app.route("/")
def home():
    return jsonify({"message": "NCF Recommendation System Active"})

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id", type=int)
    
    if user_id is None:
        return jsonify({"error": "User ID is required"}), 400
    
    # Generate personalized recommendations using NCF
    recommended_movie_ids = get_user_recommendations(user_id, model, top_n=5)
    recommendations = [{"title": movie_mapping.get(mid, "Unknown Movie")} for mid in recommended_movie_ids]
    
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True)


http://127.0.0.1:5000/recommend?user_id=1


{
  "recommendations": [
    {
      "title": "Inception"
    },
    {
      "title": "The Dark Knight"
    },
    {
      "title": "Interstellar"
    },
    {
      "title": "Fight Club"
    },
    {
      "title": "The Matrix"
    }
  ]
}