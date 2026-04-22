<div align="center">

# 🎬 Movie Recommendation Engine
### An End-to-End ML Data Pipeline Project

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://share.streamlit.io)
[![MySQL](https://img.shields.io/badge/MySQL-Database-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://mysql.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Model-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

**[🚀 Live Demo → https://movie-recommender-by-arpitasi1gh.streamlit.app/](https://movie-recommender-by-arpitasi1gh.streamlit.app/)**

</div>

---

## 📌 Problem Statement

> *How do you help a user discover their next favorite movie from a catalog of 9,700+ titles?*

This project builds a **content-based movie recommendation engine** powered by a production-style data pipeline — the same architecture used by data teams at Netflix, Swiggy, and Amazon.

---

## 🏗️ Architecture Overview

```
Raw Data (MovieLens)
      │
      ▼
 🟫 BRONZE LAYER  ──── Raw CSV data loaded into MySQL as-is
      │
      ▼
 🥈 SILVER LAYER  ──── Cleaned, formatted, null-handled data
      │
      ▼
 🥇 GOLD LAYER    ──── Enriched with avg ratings, popularity scores
      │
      ▼
 🤖 ML MODEL      ──── Cosine Similarity on Genre Matrix
      │
      ▼
 🎨 STREAMLIT APP ──── Interactive dashboard deployed on cloud
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 Smart Search | Search from 9,700+ movies instantly |
| 🤖 ML Recommendations | Content-based filtering using Cosine Similarity |
| ⭐ Live Ratings | See avg ratings & popularity per movie |
| 📊 Data Insights | Top rated & most popular movie charts |
| 🎛️ Controls | Adjust number of recommendations via slider |
| ☁️ Cloud Deployed | Accessible via public URL |

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.14 |
| **Data Processing** | Pandas, NumPy |
| **Database** | MySQL (Bronze / Silver / Gold layers) |
| **Machine Learning** | Scikit-learn (Cosine Similarity) |
| **Dashboard** | Streamlit |
| **Dataset** | MovieLens 100K (GroupLens Research) |
| **Deployment** | Streamlit Cloud |

---

## 📂 Project Structure

```
movie-recommender/
│
├── 📁 data/
│   ├── raw/          ← Bronze: original CSVs
│   ├── processed/    ← Silver: cleaned data
│   └── final/        ← Gold: analysis-ready data
│
├── 📁 notebooks/
│   ├── explore_data.ipynb     ← EDA & visualization
│   ├── load_to_mysql.ipynb    ← Bronze layer pipeline
│   ├── silver_layer.ipynb     ← Silver layer pipeline
│   ├── gold_layer.ipynb       ← Gold layer pipeline
│   └── ml_model.ipynb         ← Recommendation model
│
├── 📁 app/
│   └── app.py        ← Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## 🤖 How the ML Works

The recommendation engine uses **Content-Based Filtering**:

1. Each movie is represented as a **genre vector** (one-hot encoded)
2. **Cosine Similarity** is computed between all movie pairs
3. Given a movie, the top-N most similar movies are returned
4. Results are ranked by **average user rating**

```python
# Core ML logic (simplified)
from sklearn.metrics.pairwise import cosine_similarity

genre_matrix   = pd.get_dummies(movies['genres'])
similarity     = cosine_similarity(genre_matrix)
recommendations = similarity_df[movie].sort_values(ascending=False).head(10)
```

---

## 📊 Dataset Stats

| Metric | Value |
|---|---|
| Total Movies | 9,708 |
| Total Ratings | 100,836 |
| Total Users | 610 |
| Rating Scale | 0.5 ⭐ – 5.0 ⭐ |
| Top Rated Movie | Shawshank Redemption (4.43 ⭐) |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/arpitasi1gh/movie-recommender.git
cd movie-recommender

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
cd app
streamlit run app.py
```

---

## 🙋‍♀️ Author

**Arpita Singh**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/arpitasi1gh)

---

<div align="center">

*Built with ❤️ as part of an end-to-end data engineering + ML project*

⭐ **Star this repo if you found it useful!**

</div>
