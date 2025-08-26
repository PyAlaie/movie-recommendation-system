import config
import pandas as pd
import numpy as np 

PERCENTILE = config.BEST_MOVIES_THRESHOLD
NUMBER_OF_ROWS = config.BASELINES_ROWS

def add_weighted_ratings_charts():
    movie_metadata = pd.read_csv(config.CSV_PATH + config.MOVIE_METADATA_CSV)
    movie_metadata['genres'] = movie_metadata['genres'].apply(eval)

    print("Getting all the genres")
    genres = set()
    col = movie_metadata['genres']
    for item in col:
        for genre in item:
            genres.add(genre)

    genres.add("All")

    for genre in genres:
        print("Genre:", genre)

        if genre != "All":
            genre_spesific = movie_metadata[movie_metadata['genres'].apply(lambda x: genre in x)]
        else:
            genre_spesific = movie_metadata

        vote_counts = genre_spesific[genre_spesific['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = genre_spesific[genre_spesific['vote_average'].notnull()]['vote_average'].astype('int')
        
        C = vote_averages.mean()
        m = vote_counts.quantile(PERCENTILE)

        columns = ['title', 'release_year', 'vote_count', 'vote_average', 'popularity', 'genres', 'id', 'overview', 'poster_path']
        qualified = genre_spesific[(genre_spesific['vote_count'] >= m) & (genre_spesific['vote_count'].notnull()) & (genre_spesific['vote_average'].notnull())][columns]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')

        def weighted_rating(x):
            v = x['vote_count']
            R = x['vote_average']
            return (v/(v+m) * R) + (m/(m+v) * C)

        qualified['wr'] = qualified.apply(weighted_rating, axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(NUMBER_OF_ROWS)

        qualified.to_csv(config.BEST_RATED_PATH + '/' + genre + '.csv')


steps = {
    "add_weighted_ratings": add_weighted_ratings_charts
}

if __name__ == "__main__":
    for name, func in steps.items():
        print(name)
        func()