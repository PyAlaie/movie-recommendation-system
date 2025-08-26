import config
import pandas as pd
import shared

class HybridRecommender:
    def __init__(self, cb_model, cf_model, alpha=0.5, beta=0.5):
        self.cb = cb_model
        self.cf = cf_model
        self.alpha = alpha
        self.beta = beta


    def recommend(self, user_id=None, item_id=None, top_k=10):
        if user_id is not None:
            cf_recs = dict(self.cf.recommend(user_id, top_k*3))
            user_history = [m for m, _ in self.cf.recommend(user_id, top_k=50)]

            scores = {}
            for movie, score_cf in cf_recs.items():
                cb_boost = 0
                for hist_movie in user_history:
                    for cb_movie, cb_score in self.cb.recommand(hist_movie, k=5, title_is_id=True):
                        if cb_movie == movie:
                            cb_boost = max(cb_boost, cb_score)
                scores[movie] = self.beta*score_cf + self.alpha*cb_boost

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return ranked

        elif item_id is not None:
            cb_sims = dict(self.cb.recommand(item_id, k=top_k*2, title_is_id=True))
            cf_sims = dict(self.cf.similar_items(item_id, top_k*2))

            all_items = set(cb_sims) | set(cf_sims)
            scores = {
                movie: self.alpha*cb_sims.get(movie, 0) + self.beta*cf_sims.get(movie, 0)
                for movie in all_items
            }

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return ranked

        else:
            raise ValueError("You must provide either a user_id or an item_id.")


if __name__ == "__main__":
    from src.recommenders import collaborative_filtering, content_based

    cf = collaborative_filtering.CollabrativeFilteringMF()
    cb = content_based.ContentBasedRecommender()

    cf.fit()
    cb.build()
    
    hybrid = HybridRecommender(cb, cf, alpha=0.3, beta=0.7)
    # hybrid.recommend(user_id=1, top_k=5)

    # User-based hybrid (personalised recs)
    print("Hybrid user recs:", hybrid.recommend(user_id=1, top_k=5))

    # # Item-based hybrid (similar movies to 'Inception')
    print("Hybrid item recs:", hybrid.recommend(item_id=862, top_k=5))
