import streamlit as st
import pandas as pd
import shared, config, os 

st.title("🎬 Movie Recommendation System")

baseline, content_based, collabrative_knn, collabrative_mf, hybrid = st.tabs(["Baselines", "Content-based", "Collabrative-Filtering-KNN", "Collabrative-Filtering-MF", "Hybrid"])

with baseline:
    files = os.listdir(config.BEST_RATED_PATH)
    files = [i.split('.')[0] for i in files]

    selected_genre = st.selectbox("Genre", files)

    best_movies = pd.read_csv(config.BEST_RATED_PATH + f'/{selected_genre}.csv')
    i = 1
    for _, row in best_movies.iterrows():
        with st.expander(f"{i} . {row['title']} ({int(row['release_year'])})"):
            st.write(f"**Description:** {row['overview']}")
            st.write(f"**Genre:** {row['genres']}")
            st.write(f"**Rating:** {round(row['wr'], 2)}")
            
            i += 1

with content_based:
    from src.recommenders.content_based import ContentBasedRecommender

    model = ContentBasedRecommender()
    model.build()

    st.header("Content-Based Recommendation")

    movie_name = st.text_input("Enter a movie title:")

    if st.button("Find Similar Movies", key="cb_btn"):
        res = model.recommand(movie_name)
        results_df = shared.provide_movie_details_from_model_result(res)

        i = 1
        for _, row in results_df.iterrows():
            with st.expander(f"{i} . {row['title']}({int(row['release_year'])})"):
                st.write(f"**Description:** {row['overview']}")
                st.write(f"**Director:** {row['director']}")
                st.write(f"**Genre:** {row['genres']}")
                st.write(f"**Score:** {round(row['scores'],2)}")
                
                poster_url = "http://image.tmdb.org/t/p/original" + row['poster_path']
                st.image(poster_url, width=200)
                i += 1

with collabrative_knn:
    from src.recommenders.collaborative_filtering import CollabrativeFileteringKNN, CollabrativeFilteringMF

    knn_model = CollabrativeFileteringKNN()
    knn_model.fit()

    st.header("Collabrative-Filtering KNN Recommendation")

    movie_name2 = st.text_input("Movie Name")

    if st.button("Find Similar Movies", key="cf_btn"):

        res = knn_model.similar_items(movie_name2, item_is_id=False)
        results_df = shared.provide_movie_details_from_model_result(res)

        i = 1
        for _, row in results_df.iterrows():
            with st.expander(f"{i} . {row['title']}({int(row['release_year'])})"):
                st.write(f"**Description:** {row['overview']}")
                st.write(f"**Director:** {row['director']}")
                st.write(f"**Genre:** {row['genres']}")
                st.write(f"**Score:** {round(row['scores'],2)}")

                i += 1

with collabrative_mf:
    from src.recommenders.collaborative_filtering import CollabrativeFilteringMF

    mf_model = CollabrativeFilteringMF()
    mf_model.fit()

    st.header("Collabrative-Filtering MF Recommendation")

    user_id2 = st.text_input("User ID")

    if st.button("Find Similar Movies", key="cf_btn_mf"):

        res = mf_model.recommend(int(user_id2))
        results_df = shared.provide_movie_details_from_model_result(res)

        print(res)

        i = 1
        for _, row in results_df.iterrows():
            with st.expander(f"{i} . {row['title']}({int(row['release_year'])})"):
                st.write(f"**Description:** {row['overview']}")
                st.write(f"**Director:** {row['director']}")
                st.write(f"**Genre:** {row['genres']}")
                st.write(f"**Score:** {round(row['scores'],2)}")
                poster_url = "http://image.tmdb.org/t/p/original" + row['poster_path']
                st.image(poster_url, width=200)

                i += 1

with hybrid:
    from src.recommenders.hybrid import HybridRecommender

    from src.recommenders.collaborative_filtering import CollabrativeFilteringMF
    from src.recommenders.content_based import ContentBasedRecommender

    cf = CollabrativeFilteringMF()
    cb = ContentBasedRecommender()

    cf.fit()
    cb.build()

    hybrid_model = HybridRecommender(cb,cf)

    st.header("Hybrid Recommendation")

    item_id = st.text_input("Item ID (hybrid)")
    user_id2 = st.text_input("User ID (hybrid)")

    if st.button("Find Similar Movies", key="cf_btn_hy"):

        res = hybrid_model.recommend(user_id=int(user_id2), item_id=int(item_id))
        results_df = shared.provide_movie_details_from_model_result(res)

        print(res)

        i = 1
        for _, row in results_df.iterrows():
            with st.expander(f"{i} . {row['title']}({int(row['release_year'])})"):
                st.write(f"**Description:** {row['overview']}")
                st.write(f"**Director:** {row['director']}")
                st.write(f"**Genre:** {row['genres']}")
                st.write(f"**Score:** {round(row['scores'],2)}")
                poster_url = "http://image.tmdb.org/t/p/original" + row['poster_path']
                st.image(poster_url, width=200)

                i += 1