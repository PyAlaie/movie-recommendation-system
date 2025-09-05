import scipy.sparse as sp
import config, os
import pandas as pd
import numpy as np
import implicit
import pickle, shared
from implicit.nearest_neighbours import CosineRecommender

class CollabrativeFilteringMF:
    def __init__(self, factors=50, regularization=0.01, iterations=20, ratings_df=pd.read_csv(config.CSV_PATH + config.JOINED_RATINGS_CSV)):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations

        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations
        )

        self.ratings_df = ratings_df

        self.user_mapping = {}
        self.item_mapping = {}
        self.inv_user_mapping = {}
        self.inv_item_mapping = {}
        self.user_item_matrix = None

    def fit(self, try_to_load=True):
        if try_to_load:
            if self.try_to_load():
                print(f"Loaded!")
                return
            print("Failed loading...")
        
        users = self.ratings_df["userId"].unique()
        items = self.ratings_df["id"].unique()

        self.user_mapping = {u: i for i, u in enumerate(users)}
        self.item_mapping = {m: i for i, m in enumerate(items)}
        self.inv_user_mapping = {i: u for u, i in self.user_mapping.items()}
        self.inv_item_mapping = {i: m for m, i in self.item_mapping.items()}

        row = self.ratings_df["id"].map(self.item_mapping)
        col = self.ratings_df["userId"].map(self.user_mapping)
        data = self.ratings_df["rating"].astype(float)

        self.user_item_matrix = sp.coo_matrix(
            (data, (row, col)), 
            shape=(len(items), len(users))
        ).tocsr()

        self.item_user_matrix = sp.csr_matrix(
            (data, (row, col)), shape=(len(items), len(users))
        )

        self.user_item_matrix = self.item_user_matrix.T.tocsr()

        self.model.fit(self.user_item_matrix)

    def recommend(self, user_id, top_k=10, translate_movie_names=False):
        if user_id not in self.user_mapping:
            print(self.user_mapping)
            # print("AAAA")
            return []

        user_idx = self.user_mapping[user_id]

        item_idxs, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.user_item_matrix[user_idx],
            N=top_k
        )

        # if not translate_movie_names:
        #     return [(self.inv_item_mapping[i], s) for i, s in zip(item_idxs, scores)]
        # else:
        #     return shared.movie_title_by_id([i[0] for i in res])

        return [(self.inv_item_mapping[i], s) for i, s in zip(item_idxs, scores)]

    def similar_items(self, item_id, top_k=10):
        if item_id not in self.item_mapping:
            return []

        item_idx = self.item_mapping[item_id]
        similar, scores = self.model.similar_items(item_idx, N=top_k+1)

        similar = similar[1:]
        scores = scores[1:]
        
        return [(self.inv_item_mapping[i], s) for i, s in zip(similar, scores)]
    

    def save(self):
        model_data = {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "user_factors": self.model.user_factors,
            "item_factors": self.model.item_factors,
            "user_mapping": self.user_mapping,
            "item_mapping": self.item_mapping,
            "inv_user_mapping": self.inv_user_mapping,
            "inv_item_mapping": self.inv_item_mapping,
            "item_user_matrix": self.item_user_matrix,
            "user_item_matrix": self.user_item_matrix,
        }

        path = config.CollaborativeFilteringConfig.root_path + config.CollaborativeFilteringConfig.MFModel 

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def try_to_load(self):
        path = config.CollaborativeFilteringConfig.root_path + config.CollaborativeFilteringConfig.MFModel 
        if not os.path.exists(path):
            return False
        
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        # Recreate instance with same hyperparams
        self.model = implicit.als.AlternatingLeastSquares(
            factors=model_data["factors"],
            regularization=model_data["regularization"],
            iterations=model_data["iterations"],
        )

        # Restore trained factors
        self.model.user_factors = model_data["user_factors"]
        self.model.item_factors = model_data["item_factors"]

        # Restore mappings
        self.user_mapping = model_data["user_mapping"]
        self.item_mapping = model_data["item_mapping"]
        self.inv_user_mapping = model_data["inv_user_mapping"]
        self.inv_item_mapping = model_data["inv_item_mapping"]
        self.item_user_matrix = model_data["item_user_matrix"]
        self.user_item_matrix = model_data["user_item_matrix"]
        return True


class CollabrativeFileteringKNN:
    def __init__(self, K=20, ratings_df=pd.read_csv(config.CSV_PATH + config.JOINED_RATINGS_CSV)):
        self.K = K
        self.model = CosineRecommender(K=K)

        self.user_mapping = {}
        self.item_mapping = {}
        self.inv_user_mapping = {}
        self.inv_item_mapping = {}

        self.item_user_matrix = None
        self.user_item_matrix = None

        self.ratings_df = ratings_df

    def fit(self, try_to_load=True):
        if try_to_load:
            if self.try_to_load():
                print(f"Loaded!")
                return
            print("Failed loading...")
        
        users = self.ratings_df["userId"].unique()
        items = self.ratings_df["id"].unique()

        # Build mappings
        self.user_mapping = {u: i for i, u in enumerate(users)}
        self.item_mapping = {m: i for i, m in enumerate(items)}
        self.inv_user_mapping = {i: u for u, i in self.user_mapping.items()}
        self.inv_item_mapping = {i: m for m, i in self.item_mapping.items()}

        # Build item-user sparse matrix (rows=items, cols=users)
        row = self.ratings_df["id"].map(self.item_mapping)
        col = self.ratings_df["userId"].map(self.user_mapping)
        data = self.ratings_df["rating"].astype(float)

        self.item_user_matrix = sp.csr_matrix(
            (data, (row, col)), shape=(len(items), len(users))
        )
        self.user_item_matrix = self.item_user_matrix.T.tocsr()

        # Fit similarity model
        self.model.fit(self.item_user_matrix)

    def recommend(self, user_id, top_k=10):
        """Recommend items for a given user (item–item similarity)."""
        if user_id not in self.user_mapping:
            return []

        user_idx = self.user_mapping[user_id]

        item_idxs, scores = self.model.recommend(
            user_idx,
            self.user_item_matrix,
            N=top_k,
            filter_already_liked_items=True
        )
        return [(self.inv_item_mapping[i], s) for i, s in zip(item_idxs, scores)]

    def similar_items(self, item_id, top_k=10, item_is_id=True):
        if not item_is_id:
            item_id = shared.movie_id_by_title(item_id)

        if item_id not in self.item_mapping:
            return []

        item_idx = self.item_mapping[item_id]
        sims, scores = self.model.similar_items(item_idx, N=top_k)
        return [(self.inv_item_mapping[i], s) for i, s in zip(sims, scores)]

    def save(self):
        path = config.CollaborativeFilteringConfig.root_path + config.CollaborativeFilteringConfig.KNNModel
        
        model_data = {
            "K": self.K,
            "user_mapping": self.user_mapping,
            "item_mapping": self.item_mapping,
            "inv_user_mapping": self.inv_user_mapping,
            "inv_item_mapping": self.inv_item_mapping,
            "similarity": self.model.similarity if hasattr(self.model, "similarity") else None,
            "user_item_matrix": self.user_item_matrix,
            "item_user_matrix": self.item_user_matrix,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def try_to_load(self):
        path = config.CollaborativeFilteringConfig.root_path + config.CollaborativeFilteringConfig.KNNModel
        
        if not os.path.exists(path):
            return False
        
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.model = CosineRecommender(K=model_data["K"])
        self.user_mapping = model_data["user_mapping"]
        self.item_mapping = model_data["item_mapping"]
        self.inv_user_mapping = model_data["inv_user_mapping"]
        self.inv_item_mapping = model_data["inv_item_mapping"]
        self.item_user_matrix = model_data["item_user_matrix"]
        self.user_item_matrix = model_data["user_item_matrix"]

        self.model.fit(self.item_user_matrix)

        if model_data["similarity"] is not None:
            self.model.similarity = model_data["similarity"]

        return True


if __name__ == "__main__":
    cf = CollabrativeFilteringMF(factors=50, iterations=20)
    cf.fit()
    cf.save()

    # print("Recommendations for user 1:")
    # print(cf.recommend(user_id=1, top_k=5))

    # print("Movies similar to id=1:")
    print(cf.similar_items(item_id=862, top_k=5))

    # knn = CollabrativeFileteringKNN(K=15)
    # knn.fit(False)
    # knn.save()
    # knn.load()

    # print("Neighbourhood recommendations for user 1:")
    # print(knn.recommend(user_id=1, top_k=5))

    # print("Movies similar to id=1:")
    # print(knn.similar_items(item_id=862, top_k=5))
