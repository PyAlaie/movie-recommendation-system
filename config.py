import os

_current_file = os.path.abspath(__file__)
BASEDIR = os.path.dirname(_current_file)

# Paths
DATA_PATH = BASEDIR + '/data'
SRC_PATH = BASEDIR + '/src'
MODELS_PATH = BASEDIR + '/models'

RAW_CSV_PATH = DATA_PATH + '/raw'
CSV_PATH = DATA_PATH + '/preprocessed'
BEST_RATED_PATH = DATA_PATH + '/baselines'

# CSV
MOVIE_METADATA_CSV = "/movies_metadata.csv"
LINKS_CSV = "/links_small.csv"
RATINGS_CSV = "/ratings_small.csv"
CREDITS_CSV = "/credits.csv"
KEYWRODS_CSV = "/keywords.csv"

JOINED_MOVIES_CSV = "/movies_joined.csv"
JOINED_RATINGS_CSV = "/ratings_joined.csv"

# Variabes
BASELINES_ROWS = 250
BEST_MOVIES_THRESHOLD = 0.95

class ContentBasedConfig:
    root_path = MODELS_PATH + '/cb'
    similarity_matrix_file = "/similarity_matrix.npy"

class CollaborativeFilteringConfig:
    root_path = MODELS_PATH + '/cf'
    MFModel = "/MF.pkl"
    KNNModel = "/KNN.pkl"
