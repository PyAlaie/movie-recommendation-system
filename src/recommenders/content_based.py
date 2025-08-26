from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.sparse import hstack
from scipy import sparse
import config, os
import pandas as pd
import numpy as np
import shared

defualt_features = {
    "genres": .2,
    "keywords": .1,
    "director": .2,
    "actors": .1,
    "belongs_to_collection": .5,
    "characters": .2,
}

class ContentBasedRecommender:
    def __init__(self, features=None):
        self.movie_df = pd.read_csv(config.CSV_PATH + config.JOINED_MOVIES_CSV)
        self.vectors = None 
        self.similarity_matrix = None
        self.features = defualt_features
        self.matrix_path = config.ContentBasedConfig.root_path + config.ContentBasedConfig.similarity_matrix_file

        if isinstance(features, dict):
            self.features = features

    def extract_titles_indecies(self):
        self.titles = self.movie_df['title']
        self.title_indices = pd.Series(self.movie_df.index, index=self.movie_df['title'])
        self.id_indices = pd.Series(self.movie_df.index, index=self.movie_df['id'])

    def build(self, try_to_load=True):
        if try_to_load:
            if self.try_to_load():
                print(f"Loaded matrix from {self.matrix_path}")
                return
            print("Failed loading...")

        self.movie_df["text_features"] = self.movie_df["overview"] + " " + self.movie_df["tagline"]
        self.movie_df["text_features"].fillna("", inplace=True)
        self.movie_df["genres"].fillna("", inplace=True)
        self.movie_df["keywords"].fillna("", inplace=True)
        self.movie_df["title"].fillna("", inplace=True)
        self.movie_df["director"].fillna("", inplace=True)
        self.movie_df["actors"].fillna("", inplace=True)
        self.movie_df["characters"].fillna("", inplace=True)
        self.movie_df["production_companies"].fillna("", inplace=True)
        self.movie_df["belongs_to_collection"].fillna("", inplace=True)

        self.extract_titles_indecies()

        tfidf = TfidfVectorizer(stop_words="english")
        tdidf_matrix = tfidf.fit_transform(self.movie_df["text_features"])

        def encode_multi_hot(column_data):
            mlb = MultiLabelBinarizer()
            return mlb.fit_transform(column_data), mlb
        
        matricies = []
        for feature, weight in self.features.items():
            matrix, mlb = encode_multi_hot(self.movie_df[feature])
            matricies.append(matrix * weight)

        item_vectors = hstack([
            tdidf_matrix,
            *matricies
        ])

        self.vectors = item_vectors

        print("Calculating similarity matrix")
        print(self.vectors.shape)
        self.similarity_matrix = linear_kernel(self.vectors, self.vectors)

    def save(self):
        np.save(self.matrix_path, self.similarity_matrix)

    def try_to_load(self):
        if os.path.exists(self.matrix_path):
            self.similarity_matrix = np.load(self.matrix_path)
            self.extract_titles_indecies()
            return True
        else:
            return False

    def recommand(self, title, k=10, title_is_id=False, translate_movie_names=False):
        if not title_is_id:
            if title not in self.titles.values:
                return f"Movie {title} Not Found"

            idx = self.title_indices[title]
        else:
            idx = int(title)
            idx = self.id_indices[idx]
        
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:k+1]
        movie_indices = [i[0] for i in sim_scores]
        ids = self.movie_df.iloc[movie_indices]['id']
        titles = self.movie_df.iloc[movie_indices]['title']
        
        if not translate_movie_names:
            res = []
            for i in range(len(sim_scores)):
                res.append((ids.iloc[i], sim_scores[i][1]))
            return res
        
        else:
            res = []
            for i in range(len(sim_scores)):
                res.append((titles.iloc[i], sim_scores[i][1]))
            return res

if __name__ == "__main__":
    recommender = ContentBasedRecommender()
    recommender.build()
    recommender.save()
    res = recommender.recommand('The Avengers', title_is_id=False)
    print(res)
    # print(shared.movie_title_by_id([i[0] for i in res]))