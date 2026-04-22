import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_data():
    movies       = pd.read_csv('data/final/gold_movies.csv')
    genre_matrix = pd.read_csv('data/final/genre_matrix.csv')
    return movies, genre_matrix

movies, genre_matrix = load_data()

# ── Recommendation function (memory efficient) ────────────
def recommend_movies(movie_title, top_n=10):
    if movie_title not in genre_matrix['clean_title'].values:
        return None

    genre_cols  = genre_matrix.drop(columns=['movieId', 'clean_title'])
    movie_idx   = genre_matrix[genre_matrix['clean_title'] == movie_title].index[0]
    movie_vec   = genre_cols.iloc[movie_idx].values.reshape(1, -1)

    # Compute similarity ONLY for selected movie (not full matrix)
    similarity_scores = cosine_similarity(movie_vec, genre_cols)[0]

    scores_df = pd.DataFrame({
        'clean_title': genre_matrix['clean_title'],
        'score': similarity_scores
    })

    scores_df = scores_df[scores_df['clean_title'] != movie_title]
    top_titles = scores_df.sort_values('score', ascending=False).head(top_n)['clean_title'].tolist()

    results = movies[movies['clean_title'].isin(top_titles)].copy()
    results = results.sort_values('avg_rating', ascending=False)
    return results

# ── Header ────────────────────────────────────────────────
st.title("🎬 Movie Recommendation Engine")
st.markdown("##### Built with MovieLens data · Bronze → Silver → Gold · ML Powered")
st.divider()

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("🎛️ Controls")
top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)
st.sidebar.divider()
st.sidebar.markdown("**📊 Dataset Stats**")
st.sidebar.metric("Total Movies", f"{len(movies):,}")
st.sidebar.metric("Avg Rating", f"{movies['avg_rating'].mean():.2f} ⭐")

# ── Movie selector ────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    movie_list = sorted(movies['clean_title'].dropna().unique().tolist())
    selected   = st.selectbox("🔍 Search for a movie you like:", movie_list)

with col2:
    rating_val = movies[movies['clean_title'] == selected]['avg_rating'].values
    rating     = rating_val[0] if len(rating_val) > 0 else 0.0
    st.metric("Your Movie's Rating", f"{rating:.2f} ⭐")

st.divider()

# ── Recommendations ───────────────────────────────────────
if st.button("🚀 Get Recommendations", use_container_width=True):
    results = recommend_movies(selected, top_n)

    if results is None or results.empty:
        st.error("No recommendations found. Try another movie!")
    else:
        st.subheader(f"🍿 Because you liked **{selected}**, you'll love:")
        st.divider()

        for i, (_, row) in enumerate(results.iterrows(), 1):
            col1, col2, col3, col4 = st.columns([3, 3, 1, 1])
            with col1:
                st.markdown(f"**{i}. {row['clean_title']}**")
            with col2:
                st.caption(row['genres'])
            with col3:
                st.markdown(f"⭐ {row['avg_rating']}")
            with col4:
                year = int(row['year']) if pd.notna(row['year']) else 'N/A'
                st.caption(f"📅 {year}")
            st.divider()

# ── Charts ────────────────────────────────────────────────
st.subheader("📊 Dataset Insights")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**🏆 Top 10 Most Rated Movies**")
    top_movies = movies.nlargest(10, 'num_ratings')[['clean_title', 'num_ratings']]
    st.bar_chart(top_movies.set_index('clean_title'))

with col2:
    st.markdown("**⭐ Top 10 Highest Rated Movies (min 50 ratings)**")
    top_rated = movies[movies['num_ratings'] >= 50].nlargest(10, 'avg_rating')[['clean_title', 'avg_rating']]
    st.bar_chart(top_rated.set_index('clean_title'))
