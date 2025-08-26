import pandas as pd
import numpy as np
import config
import collaborative_filtering

def train_test_split_timeaware(ratings_df, threshold=3, test_ratio=0.5):
    ratings_df = ratings_df[ratings_df["rating"] > threshold]
    ratings_df = ratings_df.sort_values(by="timestamp")

    train, test = [], []

    for uid, group in ratings_df.groupby("userId"):
        n = len(group)
        if n < 5:  # skip very sparse users
            continue
        split_idx = int((1 - test_ratio) * n)
        train.append(group.iloc[:split_idx])
        test.append(group.iloc[split_idx:])

    return pd.concat(train), pd.concat(test)

def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / len(relevant) if relevant else 0

def hit_rate_at_k(recommended, relevant, k):
    return 1 if len(set(recommended[:k]) & set(relevant)) > 0 else 0

def evaluate_model(recommender, train_df, test_df, ks=[10, 20]):
    metrics = {k: {"precision": [], "recall": [], "hit_rate": []} for k in ks}

    for uid, group in test_df.groupby("userId"):
        test_items = group["id"].tolist()
        if not test_items:
            continue
        
        top_k = max(ks)
        recommended = recommender(uid, top_k=top_k)
        recommended = [int(i[0]) for i in recommended]
        test_items = [int(i) for i in test_items]

        if not recommended:
            continue

        for k in ks:
            prec = precision_at_k(recommended, test_items, k)
            rec = recall_at_k(recommended, test_items, k)
            hit = hit_rate_at_k(recommended, test_items, k)

            metrics[k]["precision"].append(prec)
            metrics[k]["recall"].append(rec)
            metrics[k]["hit_rate"].append(hit)

    # Average across users
    results = {k: {m: np.mean(v) for m, v in metric_dict.items()} 
               for k, metric_dict in metrics.items()}
    return results

def main():
    ratings = pd.read_csv(config.CSV_PATH + config.JOINED_RATINGS_CSV)

    train, test = train_test_split_timeaware(ratings)

    cf = collaborative_filtering.CollabrativeFilteringMF(ratings_df=train)
    cf.fit(False)

    results = evaluate_model(cf.recommend, train, test, ks=[10, 20])
    print()
    for k, res in results.items():
        print(f"Statistis for k={k}:")
        for test, number in res.items():
            print(f"  {test}: {float(number)}")
        print()

if __name__ == "__main__":
    main()
    
