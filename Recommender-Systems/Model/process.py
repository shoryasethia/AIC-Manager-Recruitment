import pandas as pd
from scipy import sparse
import streamlit as st

# Load movie titles DataFrame
movie_titles = pd.read_csv("movie_titles.csv", sep=',', header=None,
                           names=['movie_id', 'year_of_release', 'title'],
                           usecols=[0, 1, 2],
                           index_col='movie_id', encoding="ISO-8859-1")


# Initialize Streamlit app
st.set_page_config(page_title="Movie-Recommender")
st.header("Movie-Recommendation-System")

# Input movie ID
mv_id = st.text_input("Enter Movie Id: ", key="input")

# Button to suggest top 10 similar movies
submit = st.button("Suggest Top 10 Similar movies")

# with st.sidebar:
#   st.subheader("What's Happening?")
  
  # If button is clicked
if submit:
  with st.spinner("Processing..."): 
      # Load sparse matrix                  
      st.write("Loading Movie-Movie Collaborative Filtering model.")
      print("m_m_sim_sparse.npz loading started")
      
      m_m_sim_sparse = sparse.load_npz("m_m_sim_sparse.npz")
      
      st.write("Model Loaded")
      print("m_m_sim_sparse.npz loaded")
                          
      mv_id = int(mv_id)
      st.write("Movie Id Read")
      
      if mv_id in movie_titles.index:
        st.write("Movie Id is correct.")
        
        # Display movie title corresponding to given ID
        st.write("Finding Corresponding movie for given Movie Id")
        st.write(f"Movie id {mv_id} corresponds to:", movie_titles.loc[mv_id].values[1])
        
        st.write("Calculating similarities.")
        # Get similarity scores from sparse matrix
        similarities = m_m_sim_sparse[mv_id].toarray().ravel()
        similar_indices = similarities.argsort()[::-1][1:]
        similar_indices-=1 

        # Display top 10 similar movies
        st.write("Top 10 similar movies:")
        st.write(movie_titles.iloc[similar_indices[:10]])
        
      else:
        st.write("Movie Id is Incorrect.")







