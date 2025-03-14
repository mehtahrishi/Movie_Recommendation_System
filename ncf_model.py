import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load user-movie interaction data
data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
    'movie_id': [101, 102, 101, 103, 102, 104, 101, 105],
    'rating': [5, 4, 5, 3, 4, 2, 5, 4]
}
df = pd.DataFrame(data)

# Encode user and movie IDs
user_mapping = {id: idx for idx, id in enumerate(df['user_id'].unique())}
movie_mapping = {id: idx for idx, id in enumerate(df['movie_id'].unique())}
df['user_id'] = df['user_id'].map(user_mapping)
df['movie_id'] = df['movie_id'].map(movie_mapping)

# Create dataset class
class MovieDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movie_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# Define NCF Model
class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embed_size=16):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_size)
        self.movie_embedding = nn.Embedding(num_movies, embed_size)
        self.fc1 = nn.Linear(embed_size * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, user, movie):
        user_emb = self.user_embedding(user)
        movie_emb = self.movie_embedding(movie)
        x = torch.cat([user_emb, movie_emb], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

# Function to generate recommendations
def get_user_recommendations(user_id, model, top_n=5):
    model.eval()
    
    # Get user index
    user_idx = user_mapping.get(user_id, None)
    if user_idx is None:
        return []
    
    # Get all movie indices
    movie_indices = torch.arange(len(movie_mapping))
    user_tensor = torch.full((len(movie_indices),), user_idx, dtype=torch.long)

    # Predict ratings for all movies
    with torch.no_grad():
        scores = model(user_tensor, movie_indices)

    # Get top N recommendations
    top_movies = torch.argsort(scores, descending=True)[:top_n]
    recommended_movie_ids = [list(movie_mapping.keys())[i] for i in top_movies.tolist()]
    
    return recommended_movie_ids

# Prepare data
num_users = len(user_mapping)
num_movies = len(movie_mapping)
dataset = MovieDataset(df)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model, loss function, and optimizer
model = NCF(num_users, num_movies)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 5
for epoch in range(epochs):
    for users, movies, ratings in dataloader:
        optimizer.zero_grad()
        outputs = model(users, movies)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("Training complete!")

# Save the trained model
torch.save(model.state_dict(), "ncf_model.pth")
print("✅ Model saved as 'ncf_model.pth'")
