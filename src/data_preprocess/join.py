import config
from tqdm import tqdm
import pandas as pd
import os, csv

def join_movies_credits_keywords():
    movie_metadata = pd.read_csv(config.CSV_PATH + config.MOVIE_METADATA_CSV)
    credits = pd.read_csv(config.CSV_PATH + config.CREDITS_CSV)
    keywords = pd.read_csv(config.CSV_PATH + config.KEYWRODS_CSV)
    links = pd.read_csv(config.CSV_PATH + config.LINKS_CSV)

    movies_joined = movie_metadata.merge(
        credits,
        on="id",
        how="inner",
    ).merge(
        keywords,
        on="id",
        how="inner",
    ).merge(
        links,
        right_on="tmdbId",
        left_on="id",
        how="inner",
        suffixes=("_mamamia", "_kaka")
    )

    print("Credits:", credits.shape)
    print("Keywords:", keywords.shape)
    print("Movies:", movie_metadata.shape)
    print("Merged:", movies_joined.shape)

    cols = [
        'belongs_to_collection', 'genres', 'original_title', 'overview',
        'production_companies', 'production_countries', 'tagline', 'title',
        'characters', 'actors', 'director', 'keywords', 'id', 'release_year',
        'poster_path'
    ]
    
    print("Writing to file...")
    movies_joined[cols].to_csv(config.CSV_PATH + config.JOINED_MOVIES_CSV)

def join_ratings_links():
    links = pd.read_csv(config.CSV_PATH + config.LINKS_CSV)
    ratings = pd.read_csv(config.CSV_PATH + config.RATINGS_CSV)

    res = ratings.merge(
        links,
        how='inner',
        on="movieId",
    )

    res.rename(columns={"tmdbId": "id"}, inplace=True)

    res.dropna(subset=['id'], inplace=True)

    cols = ['userId', 'rating', 'timestamp', 'id']

    res = res[cols]

    print("Writing to file...")
    res.to_csv(config.CSV_PATH + config.JOINED_RATINGS_CSV)

steps = {
    "movies": join_movies_credits_keywords,
    "ratings": join_ratings_links,
}
    
if __name__ == "__main__":
    for name, func in steps.items():
        print(name)
        func()