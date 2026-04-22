import streamlit as st
import pandas as pd
import pickle

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    movies     = pd.read_csv('../data/final/gold_movies.csv')
    with open('../data/final/similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
    return movies, similarity

movies, similarity_df = load_data()

# ── Recommendation function ───────────────────────────────
def recommend_movies(movie_title, top_n=10):
    if movie_title not in similarity_df.index:
        return None
    scores     = similarity_df[movie_title].sort_values(ascending=False)
    scores     = scores.drop(movie_title, errors='ignore')
    top_titles = scores.head(top_n).index.tolist()
    results    = movies[movies['clean_title'].isin(top_titles)].copy()
    results    = results.sort_values('avg_rating', ascending=False)
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
    st.metric("Your Movie's Rating",
              f"{movies[movies['clean_title'] == selected]['avg_rating'].values[0]:.2f} ⭐")

st.divider()

# ── Show recommendations ──────────────────────────────────
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

# ── Charts section ────────────────────────────────────────
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