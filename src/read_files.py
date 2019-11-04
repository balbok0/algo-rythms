import fma_utils
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from typing import Union

genres_idx_dict = {
    'lofi': '27',
    'jazz': '4',
    'classical': '5',
    'soundtrack': '18',
    'electroacoustic': '41',
    'ambient_electro': '42',
    'alt_hip_hop': '100',
    'drum_n_bass': '337',
    'instrumental': '1235',
    'electronic': '15'
}

def fma_meta_to_csv(metadata_folder: Union[str, Path] = None, out_folder: Union[str, Path] = None):
    if metadata_folder is None:
        metadata_folder = Path('data/fma_metadata')
    elif isinstance(metadata_folder, str):
        metadata_folder = Path(metadata_folder)
    if out_folder is None:
        out_folder = Path('data/processed_data/')
    elif isinstance(out_folder, str):
        out_folder = Path(out_folder)

    if not os.path.exists(str(out_folder)):
        os.makedirs(str(out_folder))

    tracks = pd.read_csv(str(metadata_folder / 'tracks.csv'), index_col=0, header=[0, 1])
    genres = fma_utils.load(str(metadata_folder / 'genres.csv'))

    genres_filepath_dict = {
        'lofi': [],
        'jazz': [],
        'classical': [],
        'soundtrack': [],
        'electroacoustic': [],
        'ambient_electro': [],
        'alt_hip_hop': [],
        'drum_n_bass': [],
        'instrumental': [],
        'electronic': []
    }

    for key in genres_idx_dict.keys():
        genres_filepath_dict[key] = {}
        genres_filepath_dict[key]['training'] = []
        genres_filepath_dict[key]['validation'] = []
        genres_filepath_dict[key]['test'] = []


    for track_id, track_features in tqdm(tracks.iterrows(), total=len(tracks)):
        track_path = fma_utils.get_audio_path('', track_id)
        split = track_features['set', 'split']

        for genre_name, genre_idx in genres_idx_dict.items():
            if genre_idx in track_features['track', 'genres']:
                genres_filepath_dict[genre_name][split].append(track_features)

    for genre_name, genre_dict in genres_filepath_dict.items():
        np.save(
            out_folder / '{}_train.npy'.format(genre_name),
            np.array(genre_dict['training'])
        )
        np.save(
            out_folder / '{}_val.npy'.format(genre_name),
            np.array(genre_dict['validation'])
        )
        np.save(
            out_folder / '{}_test.npy'.format(genre_name),
            np.array(genre_dict['test'])
        )

if __name__ == "__main__":
    fma_meta_to_csv()
