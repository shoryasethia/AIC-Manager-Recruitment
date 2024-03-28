from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from scipy import sparse
import streamlit as st
import google.generativeai as genai
import os


# Function to ask Gemini about similarity between movies
def get_gemini_similarity(movie1, movie2):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Roughly calculate similarities between {movie1} and {movie2}.")
    return response.text

# Load movie titles DataFrame
movie_titles = pd.read_csv("movie_titles.csv", sep=',', header=None,
                           names=['movie_id', 'year_of_release', 'title'],
                           usecols=[0, 1, 2],
                           index_col='movie_id', encoding="ISO-8859-1")

# Initialize Streamlit app
st.set_page_config(page_title="Movie-Recommender")
st.header("Movie-Recommendation-System")

# Input movie ID
mv_id = st.text_input("Enter Movie Id (Refer [movie_titles.csv](https://github.com/shoryasethia/AIC-Manager-Recruitment/blob/main/Recommender-Systems/Netflix-Prize-Data/movie_titles.csv)) for id ", key="input")

# Button to suggest top 10 similar movies
submit = st.button("Suggest Top 10 Similar movies")


# If button is clicked
if submit:
    with st.spinner("Processing..."):
        # Load sparse matrix
        m_m_sim_sparse = sparse.load_npz("m_m_sim_sparse.npz")
        
        mv_id = int(mv_id)
        if mv_id in movie_titles.index:
            st.write(f"Movie id {mv_id} corresponds to:", movie_titles.loc[mv_id].values[1])
            similarities = m_m_sim_sparse[mv_id].toarray().ravel()
            similar_indices = similarities.argsort()[::-1][1:]
            similar_indices -= 1 

            st.write("Top 10 similar movies:")
            top_10_movies = movie_titles.iloc[similar_indices[:10]]
            st.write(top_10_movies)

            # Configure Gemini API
            os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            
            st.write("Calculating similarity using Google's Gemini...")
            for index, row in top_10_movies.iterrows():
                similarity_score = get_gemini_similarity(movie_titles.loc[mv_id].values[1], row['title'])
                st.write(f"Similarity score between {movie_titles.loc[mv_id].values[1]} and {row['title']}: {similarity_score}")
        else:
            st.error("Invalid movie ID. Please provide a valid movie ID.")
