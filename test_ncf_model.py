import os
import torch
from ncf_model import NCF, get_user_recommendations

# Example user ID for recommendation
user_id = 1

# Load trained model only if the file exists
if os.path.exists("ncf_model.pth"):
    model = NCF(num_users=4, num_movies=5)
    model.load_state_dict(torch.load("ncf_model.pth"))
    model.eval()

    # Get recommendations
    recommended_movies = get_user_recommendations(user_id, model, top_n=5)
    print(f"Recommended Movies for User {user_id}: {recommended_movies}")
else:
    print("❌ Model file 'ncf_model.pth' not found. Train and save the model first.")
