# Movie - Recommender System
### Based on combined effect of user-user and movie-user collaborative filtering

## Overall Implementation of movie - recommender system is divided into 3 benchmarks
### 1. Part 1 
    - Predicting Top 10 movies for a given movie 
    - Cross-checking the above list via Google’s Gemini 
[Video Demonstration](https://github.com/shoryasethia/AIC-Manager-Recruitment/blob/main/Recommender-Systems/Model/Movie-Recommender-Demo.mp4)
### 2. Part 2 
    - Validating matrix-factorization models are superior to classic 
      nearest-neighbor techniques for producing product recommendations
    - Minimizing the difference predicted and actual ratings via two 
      performance metric : RMSE and MAPE 


```
    Model             RMSE
    svd               1.0821912114098393
    knn_bsl_m         1.0821949464872673
    bsl_algo          1.0822513101982842
    knn_bsl_u         1.0822531134111517
    svdpp             1.0822543401393163
    first_algo        1.1075854286551927
    xgb_knn_bsl         1.11901091975606
    xgb_bsl           1.1234619836629212
    xgb_final         1.1234619836629212
    xgb_all_models    1.1367862682953647
    Name: rmse, dtype: object
```
### 3. Part 3
    - Using SVD matrix-factorization model for predicting Top 10 
      unwatched movies for a given user and a given watched movie
    - Comparing my SVD model via current SOTA algorithm
 [Video Demonstration](https://github.com/shoryasethia/AIC-Manager-Recruitment/blob/main/Recommender-Systems/Model/movie-recommender-for-user.mp4)

![Movie Recommendation SOTA Reference from Research Paper](https://github.com/shoryasethia/AIC-Manager-Recruitment/blob/main/Recommender-Systems/SOTA.jpg)
    
