import numpy as np
from pathlib import Path

def get_paths(genre: str, numpy: bool = False) -> (np.ndarray, np.ndarray, np.ndarray):
    path_to_genre = Path('data/processed_data') / genre
    tag = 'npy' if numpy else 'mp3'
    return np.load(path_to_genre / 'train_{}.npy'.format(tag), allow_pickle=True), \
        np.load(path_to_genre / 'val_{}.npy'.format(tag), allow_pickle=True), \
        np.load(path_to_genre / 'test_{}.npy'.format(tag), allow_pickle=True)
