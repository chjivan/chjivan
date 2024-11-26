# Step 1: Install Required Packages
# Make sure to install the following Python packages first
# pip install streamlit requests transformers sentence-transformers

import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

# Step 2: API Setup
# You need to get an API key from TMDb or IMDb for accessing movie data
TMDB_API_KEY = "2a75ac687b338532dc64516675f7ecba"

# Step 3: Define Helper Functions

def get_movie_data(query, genre_filter=None, min_rating=None, release_year=None):
    """
    Fetch movie data from TMDb based on the search query, genre, rating, and release year filters.
    """
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        movies = data.get("results", [])
        
        # Apply genre, rating, and release year filters
        if genre_filter or min_rating or release_year:
            filtered_movies = []
            for movie in movies:
                if genre_filter and genre_filter.lower() not in []:
                    continue
                if min_rating and movie.get('vote_average', 0) < min_rating:
                    continue
                if release_year and release_year != datetime.strptime(movie.get('release_date', 'N/A'), "%Y-%m-%d").year:
                    continue
                filtered_movies.append(movie)
            return filtered_movies
        return movies
    else:
        return []

# Step 4: Set Up Semantic Search Pipeline
def setup_semantic_search():
    """
    Set up a semantic search model using the sentence_transformers library.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

semantic_search_model = setup_semantic_search()

# Step 5: Define the Streamlit App Layout
st.set_page_config(page_title="Movie and Entertainment Semantic Search", layout="wide", initial_sidebar_state="expanded")
st.title("🎬 Movie and Entertainment Semantic Search")
st.write("Search for movies and entertainment content using a semantic search engine powered by NLP.")

# Sidebar filters for an enhanced user experience
st.sidebar.header("🔍 Filters")
min_rating = st.sidebar.slider("Minimum Rating", 0, 10, 5)
genre_filter = st.sidebar.text_input("Genre (e.g., Action, Comedy, Drama)")
release_year = st.sidebar.text_input("Release Year (e.g., 2020)")

# User input for the search query
user_query = st.text_input("Enter a movie or topic to search for:")

if user_query:
    # Step 6: Get Movie Data
    try:
        release_year = int(release_year) if release_year else None
    except ValueError:
        st.sidebar.error("Please enter a valid year.")
        release_year = None
    
    movie_results = get_movie_data(user_query, genre_filter, min_rating, release_year)

    if not movie_results:
        st.write("No movies found matching your search.")
    else:
        # Step 7: Show Results
        # Extract movie overviews for semantic analysis
        movie_overviews = [movie["overview"] for movie in movie_results if movie["overview"]]
        
        if movie_overviews:
            # Get semantic representations of the user's query and movie overviews
            user_vector = semantic_search_model.encode(user_query)
            overview_vectors = [semantic_search_model.encode(overview) for overview in movie_overviews]

            # Calculate cosine similarity between user query vector and movie overview vectors
            def cosine_similarity(v1, v2):
                v1, v2 = np.array(v1).flatten(), np.array(v2).flatten()
                # Ensure vectors are normalized to avoid numerical issues
                v1_norm = v1 / np.linalg.norm(v1)
                v2_norm = v2 / np.linalg.norm(v2)
                return np.dot(v1_norm, v2_norm)

            # Rank movies by similarity
            similarities = [cosine_similarity(user_vector, overview_vector) for overview_vector in overview_vectors]
            sorted_results = sorted(zip(movie_results, similarities), key=lambda x: x[1], reverse=True)

            # Display sorted results
            for movie, similarity in sorted_results:
                st.subheader(f"{movie['title']} ({movie.get('release_date', 'N/A')[:4]})")
                # Visualize rating with stars
                rating = movie.get('vote_average', 'N/A')
                if rating != 'N/A':
                    stars = '🌟' * int(rating // 2)
                    st.write(f"**Rating**: {rating} / 10 {stars}")
                else:
                    st.write(f"**Rating**: {rating} / 10")
                st.write(f"**Overview**: {movie['overview']}")
                st.write(f"**Similarity Score**: {similarity:.2f}")
                st.image(f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}", use_column_width=True)
                if movie.get('homepage'):
                    st.write(f"[More Info]({movie['homepage']})")
                st.write("---")
        else:
            st.write("No movie overviews available for semantic analysis.")

# Step 8: Run the Streamlit app
# Save this script as `app.py` and run using the command `streamlit run app.py`

