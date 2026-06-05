from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import pandas._libs.arrays as pd_arrays
import requests

app = Flask(__name__)

# OMDb API configuration
OMDB_API_KEY = '75b5969c'

# Patch pandas StringDtype for legacy pickle compatibility
_orig_stringdtype_init = pd.StringDtype.__init__

def _patched_stringdtype_init(self, *args, **kwargs):
    if len(args) > 1:
        args = args[:1]
    return _orig_stringdtype_init(self, *args, **kwargs)

pd.StringDtype.__init__ = _patched_stringdtype_init

_orig_pyxb_unpickle = pd_arrays.__pyx_unpickle_NDArrayBacked

def _patched_pyxb_unpickle(__pyx_type, __pyx_checksum, __pyx_state):
    try:
        return _orig_pyxb_unpickle(__pyx_type, __pyx_checksum, __pyx_state)
    except NotImplementedError:
        if isinstance(__pyx_state, tuple) and len(__pyx_state) >= 2:
            dtype, values = __pyx_state[0], __pyx_state[1]
            if isinstance(dtype, str) and dtype.startswith('string'):
                return pd.array(values, dtype='string')
        raise

pd_arrays.__pyx_unpickle_NDArrayBacked = _patched_pyxb_unpickle

# Load saved files
movie_pivot = pickle.load(open('movies.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
movie_list = pickle.load(open('movie_list.pkl', 'rb'))

def recommend(movie_name):

    if movie_name not in movie_pivot.index:
        return []

    movie_index = np.where(movie_pivot.index == movie_name)[0][0]
    movie_vector = movie_pivot.iloc[movie_index].values.reshape(1, -1)

    distances, indices = knn_model.kneighbors(movie_vector, n_neighbors=6)

    recommended_movies = []
    for idx in indices.flatten()[1:6]:
        recommended_movies.append(movie_pivot.index[idx])

    return recommended_movies


def _fetch_omdb_data(params):
    url = 'https://www.omdbapi.com/'
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def _strip_year_from_title(title):
    if title and title.endswith(')'):
        idx = title.rfind('(')
        if idx != -1:
            stripped = title[:idx].strip()
            if stripped:
                return stripped
    return title


def get_movie_poster(movie_name):
    if not movie_name:
        return None

    params = {
        'apikey': OMDB_API_KEY,
        't': movie_name,
        'type': 'movie'
    }
    data = _fetch_omdb_data(params)
    if data and data.get('Response') == 'True':
        poster_url = data.get('Poster')
        if poster_url and poster_url != 'N/A':
            return poster_url

    stripped_title = _strip_year_from_title(movie_name)
    if stripped_title != movie_name:
        params['t'] = stripped_title
        data = _fetch_omdb_data(params)
        if data and data.get('Response') == 'True':
            poster_url = data.get('Poster')
            if poster_url and poster_url != 'N/A':
                return poster_url

    search_params = {
        'apikey': OMDB_API_KEY,
        's': stripped_title or movie_name,
        'type': 'movie'
    }
    data = _fetch_omdb_data(search_params)
    if data and data.get('Response') == 'True':
        search_results = data.get('Search') or []
        for result in search_results:
            poster_url = result.get('Poster')
            if poster_url and poster_url != 'N/A':
                return poster_url

    return None


@app.route('/', methods=['GET', 'POST'])
def index():

    recommendations = []
    selected_movie = None
    selected_movie_poster = None

    if request.method == 'POST':

        selected_movie = request.form.get('movie')
        selected_movie_poster = get_movie_poster(selected_movie)

        recommended_titles = recommend(selected_movie)
        recommendations = [
            {
                'title': title,
                'poster': get_movie_poster(title)
            }
            for title in recommended_titles
        ]

    return render_template(
        'index.html',
        movie_list=movie_list,
        recommendations=recommendations,
        selected_movie=selected_movie,
        selected_movie_poster=selected_movie_poster
    )

if __name__ == '__main__':
    app.run(debug=True)
