from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from surprise import SVD, SVDpp, Reader, Dataset
import random
import numpy as np

# Load environment variables
load_dotenv()

# Set random seed for reproducibility
my_seed = 15
random.seed(my_seed)
np.random.seed(my_seed)

# Function to get actual and predicted ratings from Surprise predictions
def get_ratings(predictions):
    actual = np.array([pred.r_ui for pred in predictions])
    pred = np.array([pred.est for pred in predictions])
    return actual, pred

# Function to calculate RMSE and MAPE
def get_errors(predictions, print_them=False):
    actual, pred = get_ratings(predictions)
    rmse = np.sqrt(np.mean((pred - actual)**2))
    mape = np.mean(np.abs(pred - actual)/actual)
    return rmse, mape*100

# Function to train and evaluate a Surprise model
def run_surprise(algo, trainset, testset, verbose=True):
    train = dict()
    test = dict()

    # Train the model
    print('Training the model...')
    algo.fit(trainset)

    # Evaluate on train data
    print('Evaluating the model with train data..')
    train_preds = algo.test(trainset.build_testset())
    train_actual_ratings, train_pred_ratings = get_ratings(train_preds)
    train_rmse, train_mape = get_errors(train_preds)

    if verbose:
        print('-'*15)
        print('Train Data')
        print('-'*15)
        print("RMSE : {}\n\nMAPE : {}\n".format(train_rmse, train_mape))

    train['rmse'] = train_rmse
    train['mape'] = train_mape
    train['predictions'] = train_pred_ratings

    # Evaluate on test data
    print('\nEvaluating for test data...')
    test_preds = algo.test(testset)
    test_actual_ratings, test_pred_ratings = get_ratings(test_preds)
    test_rmse, test_mape = get_errors(test_preds)

    if verbose:
        print('-'*15)
        print('Test Data')
        print('-'*15)
        print("RMSE : {}\n\nMAPE : {}\n".format(test_rmse, test_mape))

    test['rmse'] = test_rmse
    test['mape'] = test_mape
    test['predictions'] = test_pred_ratings

    print('\n'+'-'*45)

    return train, test

# Initialize Streamlit app
st.set_page_config(page_title="Movie-Recommender")
st.header("Movie Recommender for a User")

# Input user and movie ID 
u = st.text_input("Enter User Id (Refer [this](https://raw.githubusercontent.com/shoryasethia/AIC-Manager-Recruitment/main/Recommender-Systems/sample/small/reg_train.csv?token=GHSAT0AAAAAACPHZ2KRC7UXRRCDHK3VMXJSZQHDNUA)) ", key="user_input")
m = st.text_input("Enter MovieId (Refer [this](https://raw.githubusercontent.com/shoryasethia/AIC-Manager-Recruitment/main/Recommender-Systems/sample/small/reg_train.csv?token=GHSAT0AAAAAACPHZ2KRC7UXRRCDHK3VMXJSZQHDNUA)) ", key="movie_input")

# Button to suggest top 10 similar movies
submit = st.button("Recommend Top 10 movies to User")

# If button is clicked
if submit:
    with st.spinner("Processing..."):
        reg_train = pd.read_csv('reg_train.csv', names = ['user', 'movie', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4', 'sur5','smr1', 'smr2', 'smr3', 'smr4', 'smr5', 'UAvg', 'MAvg', 'rating'], header=None)
        reg_test_df = pd.read_csv('reg_test.csv', names = ['user', 'movie', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4', 'sur5',
                                                          'smr1', 'smr2', 'smr3', 'smr4', 'smr5',
                                                          'UAvg', 'MAvg', 'rating'], header=None)
        
        reader = Reader(rating_scale=(1,5))

        train_data = Dataset.load_from_df(reg_train[['user', 'movie', 'rating']], reader)
        trainset = train_data.build_full_trainset() 
        testset = list(zip(reg_test_df.user.values, reg_test_df.movie.values, reg_test_df.rating.values))
        
        # Initialize the model
        svd = SVD(n_factors=100, biased=True, random_state=30, verbose=False)
        svd_train_results, svd_test_results = run_surprise(svd, trainset, testset, verbose=False)
        
        # Initialize the model
        svdpp = SVDpp(n_factors=50, random_state=30, verbose=False)
        svdpp_train_results, svdpp_test_results = run_surprise(svdpp, trainset, testset, verbose=False)

        movie_id = m
        user_id = u
        all_movie_ids = list(trainset.all_items())
        
        try:
            all_movie_ids.remove(movie_id)
        except ValueError:
            pass
        
        # Predict ratings for all movies for the given user via SVD
        predicted_ratings = [svd.predict(user_id, movie_id).est for movie_id in all_movie_ids]

        # Combine movie IDs with predicted ratings
        movie_ratings = list(zip(all_movie_ids, predicted_ratings))

        # Sort the movies based on predicted ratings in descending order
        sorted_movie_ratings = sorted(movie_ratings, key=lambda x: x[1], reverse=True)

        # Get the top 5 recommended movies
        top_5_recommendations = sorted_movie_ratings[:10]

        # Display the top 5 recommendations
        for movie_id, predicted_rating in top_5_recommendations:
            st.write("Movie ID:", movie_id, "| Predicted Rating:", predicted_rating)
