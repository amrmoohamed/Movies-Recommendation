from PyMovieDb import IMDB
import json
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
from surprise.reader import Reader
from surprise.dataset import Dataset
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.accuracy import rmse,mse,mae
from surprise.dump import dump,load


def add_poster(x):
    imdb = IMDB()
    res = imdb.get_by_id(x)
    if res =='{"status": 404, "message": "No Result Found!", "result_count": 0, "results": []}': 
          print("error")
          return 'https://icon-library.com/images/404-error-icon/404-error-icon-24.jpg'
    else:
          data= json.loads(res)
          print(data["poster"])
          return data["poster"]


def roundd(rating):
    rating = float(rating)
    if rating - int(rating) > 0.75:
        rating = int(rating) + 1
    elif rating - int(rating) < 0.75 and (rating - int(rating) > 0.5) :
        rating = int(rating) + 0.5
    elif rating - int(rating) > 0.25 and (rating - int(rating) < 0.5) :
        rating=int(rating) + 0.5
    elif rating - int(rating) < 0.25:
        rating = int(rating)
    else:
        rating = int(rating)
    return rating

def add_tt(x):
    x=str(x)
    reminders=7-len(x)
    tconst= f"tt{reminders*'0'}{x}"
    return tconst

def recommend_top_k_movies(user_id, K=100):
    predictions, algo = load('./models/svd')
    # Load ratings dataset
    rating = pd.read_csv('./data/ratings.csv')
    # Load movies dataset
    movie = pd.read_csv('./data/movies.csv')
    links=pd.read_csv('./data/links.csv')
    # Merge two datasets to have a better picture
    df = pd.merge(rating, movie, on='movieId')
    utility_matrix = df.pivot(index='userId', columns='movieId', values='rating')
    utility_matrix = utility_matrix.fillna(0)
    user_idx = np.where(utility_matrix.index == user_id)[0][0]
    candidates = utility_matrix.iloc[user_idx][utility_matrix.iloc[user_idx] == 0]

    test = []
    for rating, index in zip(candidates, candidates.index):
        test.append((user_id, index, rating))
    predictions = algo.test(test)

    items = np.array([item.iid for item in predictions]).reshape(-1, 1)
    predicted = np.array([item.est for item in predictions]).reshape(-1, 1)
    real = np.array([item.r_ui for item in predictions]).reshape(-1, 1)

    modified_predicted = np.concatenate([items, predicted], axis=1)
    df_biased = pd.DataFrame(modified_predicted, columns=['movieId', 'rating'])
    df_sorted = df_biased.sort_values('rating', ascending=False)
    df_sorted['rating'] = df_sorted['rating'].apply(roundd)
    df_sorted['movieId'] = df_sorted['movieId'].astype(float).astype(int)

    # Merge with movies DataFrame
    df_merged = pd.merge(df_sorted, movie[['movieId', 'title']], on='movieId')
    df_merged_links = pd.merge(df_merged, links[['movieId', 'imdbId']], on='movieId')
    df_merged_links["imdbId"]=df_merged_links["imdbId"].apply(add_tt)
    # Drop rows with NaN values
    df_merged_links = df_merged_links.dropna().head(K)

    return df_merged_links



def recommend_similar_movies(similarity_matrix,movie_id, K=100):
    movie_index = movie_id - 1
    similarity_scores = similarity_matrix[movie_index]
    top_K_indices = np.argsort(similarity_scores)[::-1][:K]
    top_K_movies = [(index + 1, similarity_scores[index]) for index in top_K_indices]
    
    # Convert the list of tuples to a DataFrame
    df = pd.DataFrame(top_K_movies, columns=["movieId", "similarity"])
    df["similarity"]=df["similarity"]*100
    df["similarity"]=df["similarity"].apply(lambda x: round(x, 2))
    
    return df

# print(lst)