import config 
import pandas as pd

def movie_title_by_id(ids):
    movies = pd.read_csv(config.CSV_PATH + config.MOVIE_METADATA_CSV)
    res = movies[movies['id'].isin(ids)]
    res = res.set_index('id').loc[ids].reset_index()
    return res['title']

def movie_id_by_title(title):
    movies = pd.read_csv(config.CSV_PATH + config.MOVIE_METADATA_CSV)
    res = movies[movies['title'] == title]
    res = list(res['id'])
    res = res[0]

    return res

def provide_movie_details_from_model_result(result:list):
    movies = pd.read_csv(config.CSV_PATH + config.JOINED_MOVIES_CSV)

    ids = [i[0] for i in result]
    scores = [i[1] for i in result]

    cols = [
        'belongs_to_collection', 'genres', 'title',
        'overview', 'actors', 'director', 'release_year',
        'poster_path'
    ]

    movies = movies[movies['id'].isin(ids)]
    movies = movies.set_index('id').loc[ids].reset_index()
    movies = movies[cols]
    movies['scores'] = scores

    return movies
