# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# --- App Title ---
st.title("🎬 Movie Dashboard")

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv('/workspaces/movie-dashboard/data/movie_ratings.csv')
    df['genres'] = df['genres'].str.split('|')
    df_exploded = df.explode('genres')
    return df_exploded

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")
age_filter = st.sidebar.slider(
    "Select Age Range",
    int(df['age'].min()),
    int(df['age'].max()),
    (18, 60)
)
gender_filter = st.sidebar.multiselect(
    "Select Gender",
    df['gender'].unique(),
    default=list(df['gender'].unique())
)
genre_filter = st.sidebar.multiselect(
    "Select Genres",
    df['genres'].unique(),
    default=list(df['genres'].unique())
)

# Apply filters
df_filtered = df[
    (df['age'] >= age_filter[0]) &
    (df['age'] <= age_filter[1]) &
    (df['gender'].isin(gender_filter)) &
    (df['genres'].isin(genre_filter))
]

st.subheader("Filtered Dataset Preview")
st.dataframe(df_filtered.head(10))

# --- 1️⃣ Genre Breakdown ---
st.subheader("Number of Ratings per Genre")
genre_counts = df_filtered['genres'].value_counts()
plt.figure(figsize=(12,6))
sns.set_theme(style="whitegrid")
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="viridis")
plt.xticks(rotation=45, ha='right')
plt.xlabel("Genre")
plt.ylabel("Number of Ratings")
plt.title("Number of Ratings per Genre")
for i, v in enumerate(genre_counts.values):
    plt.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=10)
st.pyplot(plt.gcf())

# --- 2️⃣ Highest Viewer Satisfaction by Genre ---
st.subheader("Average Rating per Genre")
genre_mean_rating = df_filtered.groupby('genres')['rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=genre_mean_rating.index, y=genre_mean_rating.values, palette="coolwarm")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Average Rating")
plt.xlabel("Genre")
plt.title("Average Viewer Rating per Genre")
for i, v in enumerate(genre_mean_rating.values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
st.pyplot(plt.gcf())

# --- 3️⃣ Mean Rating vs Viewer Age for Selected Genres ---
st.subheader("Rating vs Viewer Age for Selected Genres")
selected_genres = st.multiselect(
    "Select up to 4 Genres",
    df['genres'].unique(),
    default=['Action','Comedy','Drama','Romance']
)
if selected_genres:
    df_age_genre = df_filtered[df_filtered['genres'].isin(selected_genres)]
    df_age_mean = df_age_genre.groupby(['age','genres'])['rating'].mean().reset_index()
    plt.figure(figsize=(12,6))
    sns.set_theme(style="white")
    sns.lineplot(
        data=df_age_mean,
        x='age',
        y='rating',
        hue='genres',
        palette="Set2",
        marker="o",
        linewidth=3,
        markersize=8
    )
    plt.xlabel("Viewer Age")
    plt.ylabel("Average Rating")
    plt.title("Average Rating vs Age per Genre")
    plt.legend(title='Genre', fontsize=10, title_fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(np.arange(1, 5.5, 0.5))
    plt.ylim(1, 5)
    st.pyplot(plt.gcf())

# --- 4️⃣ Mean Rating per Genre with Number of Ratings ---
st.subheader("Mean Rating per Genre (Number of Ratings Shown)")
genre_stats = df_filtered.groupby('genres').agg(
    num_ratings=('rating','count'),
    mean_rating=('rating','mean')
).sort_values('mean_rating', ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=genre_stats.index, y=genre_stats['mean_rating'], palette="magma")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean Rating")
plt.xlabel("Genre")
plt.title("Mean Rating per Genre")
for i, v in enumerate(genre_stats['mean_rating']):
    plt.text(i, v + 0.02, str(genre_stats['num_ratings'].iloc[i]), ha='center', va='bottom', fontsize=10)
st.pyplot(plt.gcf())

# --- 5️⃣ Best-Rated Movies (Min 50 Ratings) ---
st.subheader("Top 5 Best-Rated Movies (≥50 Ratings)")
movie_stats = df_filtered.groupby('title').agg(count=('rating','count'), mean_rating=('rating','mean'))
top_5_50 = movie_stats[movie_stats['count'] >= 50].sort_values('mean_rating', ascending=False).head(5)
plt.figure(figsize=(12,6))
sns.barplot(x=top_5_50.index, y=top_5_50['mean_rating'], palette="coolwarm")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean Rating")
plt.xlabel("Movie Title")
plt.title("Top 5 Best-Rated Movies (≥50 Ratings)")
for i, v in enumerate(top_5_50['mean_rating']):
    plt.text(i, v + 0.02, str(top_5_50['count'].iloc[i]), ha='center', va='bottom', fontsize=10)
st.pyplot(plt.gcf())

# --- 6️⃣ Best-Rated Movies (Min 150 Ratings) ---
st.subheader("Top 5 Best-Rated Movies (≥150 Ratings)")
top_5_150 = movie_stats[movie_stats['count'] >= 150].sort_values('mean_rating', ascending=False).head(5)
plt.figure(figsize=(12,6))
sns.barplot(x=top_5_150.index, y=top_5_150['mean_rating'], palette="magma")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean Rating")
plt.xlabel("Movie Title")
plt.title("Top 5 Best-Rated Movies (≥150 Ratings)")
for i, v in enumerate(top_5_150['mean_rating']):
    plt.text(i, v + 0.02, str(top_5_150['count'].iloc[i]), ha='center', va='bottom', fontsize=10)
st.pyplot(plt.gcf())
