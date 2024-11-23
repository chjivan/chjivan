# Save this file with a .py extension, e.g., semantic_search_app.py
import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Step 1: Data Collection
# Use RapidAPI to collect or manually create a dataset.
# Here, we use a simple list of sample documents for the demo purpose.
documents = [
    "Streamlit is an open-source Python library for creating web apps for machine learning and data science.",
    "RapidAPI is an API marketplace for developers to find, connect, and manage APIs.",
    "BERTopic is a topic modeling technique that uses transformers to generate topics.",
    "Python is a popular programming language for data science and web development.",
    "Transformers are deep learning models used primarily for natural language processing."
]

data = pd.DataFrame({"Document": documents})

# Step 2: User Interface
st.title("Semantic Search AI Application")
st.write("This is a simple semantic search application using Streamlit, TF-IDF, and cosine similarity.")

# Step 3: User Query
query = st.text_input("Enter your search query:")

# Step 4: Data Analysis - Compute Semantic Similarity
if query:
    # Optional: Expand dataset with Wikipedia API results based on the search query
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        search_results = response.json()["query"]["search"]
        for result in search_results:
            documents.append(result["snippet"])
    
    # Re-create the DataFrame with updated documents
    data = pd.DataFrame({"Document": documents})

    # Calculate TF-IDF for both the documents and the query
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data["Document"].tolist() + [query])

    # Compute cosine similarity between the query vector and document vectors
    cosine_similarities = cosine_similarity(vectors[-1:], vectors[:-1])

    # Step 5: Data Presentation
    st.subheader("Search Results:")
    results = pd.DataFrame({"Document": data["Document"], "Similarity": cosine_similarities.flatten()})
    results = results.sort_values(by="Similarity", ascending=False)
    for index, row in results.iterrows():
        st.write(f"- **Similarity Score**: {row['Similarity']:.2f}")
        st.write(f"- **Document**: {row['Document']}")
        st.write("---")

