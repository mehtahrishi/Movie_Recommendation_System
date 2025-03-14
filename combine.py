import pandas as pd

# Load processed dataset
movies = pd.read_csv("processed_movies.csv")

# Function to clean and combine features
def combine_features(row):
    try:
        genres = " ".join(eval(row['genres'])) if isinstance(row['genres'], str) else ""
        keywords = " ".join(eval(row['keywords'])) if isinstance(row['keywords'], str) else ""
        cast = " ".join(eval(row['cast'])) if isinstance(row['cast'], str) else ""
        overview = row['overview'] if isinstance(row['overview'], str) else ""
        return f"{genres} {keywords} {cast} {overview}"
    except:
        return ""

# Apply function to create 'combined_features' column
movies['combined_features'] = movies.apply(combine_features, axis=1)

# Save the updated dataset
movies.to_csv("processed_movies.csv", index=False)
print("✅ Cleaned and fixed 'combined_features'. Try running recommendation.py again!")
