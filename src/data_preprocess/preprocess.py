import config
import pandas as pd
import ast

def safe_literal_eval(string_value):
    try:
        return ast.literal_eval(string_value)
    except (ValueError, SyntaxError):
        return None

def extract_collection_name(collection_dict, field='name'):
    if isinstance(collection_dict, dict) and field in collection_dict:
        return collection_dict[field]
    return None

def extract_names(genre_list, field='name'):
    if isinstance(genre_list, list):
        return [genre[field] for genre in genre_list]
    return []

def extract_director_name(crew_dict):
    for crew in crew_dict:
        if isinstance(crew, dict) and 'job' in crew.keys():
            res = crew.get('job', None)
            if res == "Director":
                return crew.get('name', None)
    return None

def extract_top_k_casts(casts, field='name', k=4):
    if isinstance(casts, list):
        return [cast[field] for cast in casts][:k]
    return []

def preprocess_movie_metadata():
    movie_metadata_raw = pd.read_csv(config.RAW_CSV_PATH + config.MOVIE_METADATA_CSV)

    movie_metadata_raw.drop_duplicates(subset="id", inplace=True)

    # Removig adults that are not true nor flase
    movie_metadata_raw = movie_metadata_raw[movie_metadata_raw['adult'].isin(['True', 'False'])]
    movie_metadata_raw['adult'] = movie_metadata_raw['adult'].astype(bool)

    # Parse json objects
    movie_metadata_raw['belongs_to_collection'] = movie_metadata_raw['belongs_to_collection'].apply(safe_literal_eval).apply(extract_collection_name)
    movie_metadata_raw['genres'] = movie_metadata_raw['genres'].apply(safe_literal_eval).apply(extract_names)
    movie_metadata_raw['production_companies'] = movie_metadata_raw['production_companies'].apply(safe_literal_eval).apply(extract_names)
    movie_metadata_raw['production_countries'] = movie_metadata_raw['production_countries'].apply(safe_literal_eval).apply(extract_names)
    movie_metadata_raw['spoken_languages'] = movie_metadata_raw['spoken_languages'].apply(safe_literal_eval).apply(extract_names)

    # Parse datetime
    movie_metadata_raw['release_date'] = pd.to_datetime(movie_metadata_raw['release_date'], errors='coerce')
    movie_metadata_raw['release_year'] = movie_metadata_raw['release_date'].dt.year
    movie_metadata_raw['release_month'] = movie_metadata_raw['release_date'].dt.month
    movie_metadata_raw['release_day'] = movie_metadata_raw['release_date'].dt.day

    # Cast some cols
    for col in ["popularity", "budget", "id"]:
        movie_metadata_raw[col] = pd.to_numeric(movie_metadata_raw[col], errors='coerce')
    
    movie_metadata_raw.drop(["homepage", "imdb_id", "video", "status"], axis=1, inplace=True)

    movie_metadata_raw.to_csv(config.CSV_PATH + config.MOVIE_METADATA_CSV)


def preprocess_links():
    links_raw = pd.read_csv(config.RAW_CSV_PATH + config.LINKS_CSV)

    links_raw.drop_duplicates(subset="tmdbId", inplace=True)

    links_raw.drop(["imdbId"], axis=1, inplace=True)

    links_raw.to_csv(config.CSV_PATH + config.LINKS_CSV)


def preprocess_ratings():
    ratings_raw = pd.read_csv(config.RAW_CSV_PATH + config.RATINGS_CSV)

    ratings_raw.to_csv(config.CSV_PATH + config.RATINGS_CSV)


def preprocess_keywrods():
    keywords_raw = pd.read_csv(config.RAW_CSV_PATH + config.KEYWRODS_CSV)

    keywords_raw.drop_duplicates(subset="id", inplace=True)

    keywords_raw['keywords'] = keywords_raw['keywords'].apply(safe_literal_eval).apply(extract_names)

    keywords_raw.to_csv(config.CSV_PATH + config.KEYWRODS_CSV)


def preprocess_credits():
    credits_raw = pd.read_csv(config.RAW_CSV_PATH + config.CREDITS_CSV)

    credits_raw.drop_duplicates(subset='id', inplace=True)

    credits_raw['characters'] = credits_raw['cast'].apply(safe_literal_eval).apply(lambda x: extract_top_k_casts(x, 'character'))
    credits_raw['actors'] = credits_raw['cast'].apply(safe_literal_eval).apply(lambda x: extract_top_k_casts(x, 'name'))
    credits_raw['director'] = credits_raw['crew'].apply(safe_literal_eval).apply(lambda x: extract_director_name(x))
    # TODO: parse other crew memebers too, like screenwriter and music producer
    
    credits_raw.drop(["cast", "crew"], axis=1, inplace=True)

    credits_raw.to_csv(config.CSV_PATH + config.CREDITS_CSV)

steps = {
    "movies": preprocess_movie_metadata,
    "links": preprocess_links,
    "ratings": preprocess_ratings,
    "keywords": preprocess_keywrods,
    "credits": preprocess_credits,
}

if __name__ == "__main__":
    for name, func in steps.items():
        print(name)
        func()