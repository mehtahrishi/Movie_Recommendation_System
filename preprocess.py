import pandas as pd
import ast

def load_and_merge_data():
    # Load datasets with proper encoding handling
    movies = pd.read_csv("tmdb_5000_movies.csv", encoding="utf-8")
    credits = pd.read_csv("tmdb_5000_credits.csv", encoding="utf-8")

    # Rename 'movie_id' in credits to match 'id' in movies
    credits.rename(columns={"movie_id": "id"}, inplace=True)

    # Merge datasets on 'id', keeping only necessary columns
    movies = movies.merge(credits[['id', 'cast']], on="id", how="left")

    # Print merged column names to verify correct merging
    print("Merged DataFrame Columns:", movies.columns)

    return movies

def clean_data(movies):
    # Ensure the required columns exist before selecting
    required_columns = ['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'vote_average', 'budget', 'revenue']
    available_columns = list(movies.columns)

    # Find missing columns (if any)
    missing_columns = [col for col in required_columns if col not in available_columns]
    if missing_columns:
        print(f"⚠️ Warning: Missing columns in dataset: {missing_columns}")
        return None

    # Select required columns and create a copy to avoid warnings
    movies = movies[required_columns].copy()

    # Drop rows with missing values
    movies.dropna(inplace=True)

    # Convert JSON-like columns (genres, keywords, cast) to readable format
    def convert(obj):
        try:
            return [i['name'] for i in ast.literal_eval(obj)]
        except:
            return []

    movies.loc[:, 'genres'] = movies['genres'].apply(convert)
    movies.loc[:, 'keywords'] = movies['keywords'].apply(convert)
    
    # Extract only main 3 cast members
    def convert_cast(obj):
        try:
            return [i['name'] for i in ast.literal_eval(obj)[:3]]
        except:
            return []

    movies.loc[:, 'cast'] = movies['cast'].apply(convert_cast)

    return movies

def main():
    print("Loading and merging data...")
    movies = load_and_merge_data()

    if movies is None:
        print("❌ Data loading failed. Exiting.")
        return
    
    print("Cleaning data...")
    movies = clean_data(movies)

    if movies is None:
        print("❌ Data cleaning failed. Exiting.")
        return

    # Save processed data
    movies.to_csv("processed_movies.csv", index=False)
    print("✅ Data preprocessing complete! Saved as 'processed_movies.csv'.")

if __name__ == "__main__":
    main()
