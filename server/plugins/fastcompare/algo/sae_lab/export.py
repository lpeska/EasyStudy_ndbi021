"""Portable, framework-free artifacts for student SAE exploration."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def export_sae_artifacts(loader, codes, popularity, destination, variant='abs'):
    """Write aligned SAE codes and movie metadata without serializing EasyStudy objects.

    destination is a directory. Called before the adapter discards its loader.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    codes = np.asarray(codes, dtype=np.float32)
    popularity = np.asarray(popularity, dtype=np.float32).reshape(-1)
    if codes.ndim != 2 or codes.shape[0] != len(popularity):
        raise ValueError('SAE codes and popularity have incompatible shapes')
    n = len(popularity)
    metadata = loader.items_df.copy()
    if not {'item_id', 'title'}.issubset(metadata.columns):
        raise ValueError('Loader items_df requires item_id and title')
    if metadata.item_id.duplicated().any():
        raise ValueError('Duplicate item_id in items_df')
    metadata = metadata.set_index('item_id', drop=False)
    movie_ids = [loader.get_item_id(i) for i in range(n)]
    if len(set(movie_ids)) != n or not set(movie_ids).issubset(metadata.index):
        raise ValueError('Invalid row-to-MovieLens-ID mapping')
    movies = metadata.loc[movie_ids, ['item_id', 'title'] + (["genres"] if 'genres' in metadata.columns else [])].copy()
    movies.insert(0, 'item_index', np.arange(n))
    movies['popularity'] = popularity.astype(np.int64)
    if not np.all(popularity == movies['popularity'].to_numpy()):
        raise ValueError('Popularity contains noninteger values')
    np.savez_compressed(destination / 'sae_codes.npz', codes=codes, popularity=popularity)
    movies.to_csv(destination / 'sae_movies.csv', index=False, encoding='utf-8')
    (destination / 'sae_metadata.json').write_text(json.dumps({
        'variant': variant, 'n_items': n, 'n_features': codes.shape[1],
        'codes': 'sae_codes.npz', 'movies': 'sae_movies.csv',
        'popularity_definition': 'number of training interactions',
    }, indent=2), encoding='utf-8')
    return destination
