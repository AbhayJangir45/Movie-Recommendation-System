import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

# Load ratings data
user_data = pd.read_csv(
    'u.data',
    sep='\t',
    usecols=[0, 1, 2],
    names=['user_id', 'movie_id', 'rating']
)

# Load movie metadata
column_names = [
    'movie_id', 'movie title', 'release date', 'video release date',
    'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
    "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
    'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

movie_profile = pd.read_csv(
    'u.item',
    sep='|',
    encoding='latin-1',
    header=None,
    names=column_names,
    usecols=[0, 1]
)

# Merge datasets
final_data = pd.merge(movie_profile, user_data, on='movie_id')

# Create movie-user matrix
profile = final_data.pivot_table(
    values='rating',
    index='movie title',
    columns='user_id'
).fillna(0)

# Train NearestNeighbors model
model = NearestNeighbors(n_neighbors=4, metric='cosine', algorithm='brute')
model.fit(profile)

# Save trained model and data
pickle.dump(model, open('knn_model.pkl', 'wb'))
pickle.dump(profile, open('movies.pkl', 'wb'))
pickle.dump(profile.index.tolist(), open('movie_list.pkl', 'wb'))

print('Model trained successfully!')
