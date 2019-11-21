import fma_utils

import numpy as np
import pandas as pd
import librosa as lbs
from audioread.exceptions import NoBackendError

from tqdm import tqdm

import glob
import os
from pathlib import Path
from shutil import move
from zipfile import ZipFile

from typing import Union

# Disable warnings, because librosa is annoying
import warnings
warnings.filterwarnings("ignore")

# All the genes being considered for the training task
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
                genres_filepath_dict[genre_name][split].append(track_path)

    for genre_name, genre_dict in genres_filepath_dict.items():
        genre_dir = out_folder / genre_name
        if not os.path.exists(genre_dir):
            os.makedirs(genre_dir)
        print('genre_name: {}'.format(genre_name))
        np.save(
            genre_dir / 'train_mp3.npy',
            np.unique(genre_dict['training'])
        )
        np.save(
            genre_dir / 'val_mp3.npy',
            np.unique(genre_dict['validation'])
        )
        np.save(
            genre_dir / 'test_mp3.npy',
            np.unique(genre_dict['test'])
        )


def get_paths(genre: str, numpy: bool = False) -> (np.ndarray, np.ndarray, np.ndarray):
    path_to_genre = Path('data/processed_data') / genre
    tag = 'npy' if numpy else 'mp3'
    return np.load(path_to_genre / 'train_{}.npy'.format(tag), allow_pickle=True), \
        np.load(path_to_genre / 'val_{}.npy'.format(tag), allow_pickle=True), \
        np.load(path_to_genre / 'test_{}.npy'.format(tag), allow_pickle=True)


def total_number_of_tracks():
    # Returns total number of tracks across all genres.
    result = 0
    paths = glob.glob("data/processed_data/**/*.npy")

    for p in paths:
        result += len(np.load(p))
    return result


def mp3_to_npy(genre: str, path_read: Path = None, path_save_arr: Path = None, path_save_paths: Path = None):
    """ Given a genre takes all of it's songs,
    extract them to the correct folder and
    converts to npy arrays representing spectrograms.

    Arguments:
        genre {str} -- Songs from what genre to convert.
        path_read {Path} -- Path to zip file containing songs.  (default: 'data/fma_metadata')
        path_save_arr {Path} -- Path to where to save npy arrays.  (default: 'data/npy_data')
        path_save_paths {Path} --
            Path to where to save arrays containing paths to the spectrogram .npy files.
            (default: 'data/processed_data')
    """
    # Convert default Nones to actual defaults
    if path_read is None:
        path_read = Path('data/fma_metadata')
    if path_save_arr is None:
        path_save_arr = Path('data/npy_data')
    if path_save_paths is None:
        path_save_paths = Path('data/processed_data')

    # Make dictionaries to save data to
    if not os.path.exists(path_save_paths):
        os.makedirs(path_save_paths)
    if not os.path.exists(path_save_paths):
        os.makedirs(path_save_paths)

    # Decompress required files
    train, val, test = get_paths(genre)

    # Make arrays of paths to the newly created npy files
    train_paths, val_paths, test_paths = [], [], []

    with ZipFile(path_read, mode='r') as zipped:
        print("unzipping {}".format(path_read))

        # Do the conversion for all datasets.
        for name, dataset in [('Train', train), ('Validation', val), ('Test', test)]:
            # Loop over files in dataset and convert to npy
            for file in tqdm(dataset, desc="{}:".format(name)):
                file_npy = file[:-4] + '.npy'  # Numpy file to be created
                file_folder = Path(file).parent  # Folder to which extract .mp3 file

                if not os.path.exists(path_save_arr / file_npy):  # Only process file once
                    # Extract .mp3 file and then move it to the right directory.
                    extract_path = zipped.extract("fma_large/{}".format(file))
                    if not os.path.exists(extract_path):  # Unsuccessful extract? Continue
                        continue

                    try:
                        # Read newly .mp3 created file
                        y, sr = lbs.load(extract_path)
                        # Convert it to spectrogram numpy array
                        D = lbs.amplitude_to_db(np.abs(lbs.stft(y)), ref=np.max)
                    except (RuntimeError, NoBackendError):
                        continue
                    # Save spectrogram to correct folder (optionally creating it)
                    if not os.path.exists(path_save_arr / file_folder):
                        os.makedirs(path_save_arr / file_folder)
                    np.save(path_save_arr / file_npy, D)

                    # Remove .mp3 file
                    os.remove(extract_path)

                # Append paths to npy files
                if name == 'Train':
                    train_paths.append(path_save_arr / file_npy)
                elif name == 'Validation':
                    val_paths.append(path_save_arr / file_npy)
                elif name == 'Test':
                    test_paths.append(path_save_arr / file_npy)


    # Save paths to new data
    np.save(path_save_paths / genre / 'train_npy.npy', train_paths)
    np.save(path_save_paths / genre / 'val_npy.npy', val_paths)
    np.save(path_save_paths / genre / 'test_npy.npy', test_paths)


def check_data():
    """Temporary function just used to check various stuff we created.
    """
    for g in genres_idx_dict.keys():
        print('Genre {}\tSize: {}'.format(g, sum([len(x) for x in get_paths(g)])))
    g = 'instrumental'
    print('Genre {}\tSize: {}'.format(g, sum([len(x) for x in get_paths(g, numpy=True)])))

    for p in glob.glob('data/npy_data/**/*.npy'):
        print(np.load(p).shape)


if __name__ == "__main__":
    pass

    # First step get paths from tracks and genres and make numpy arrays
    # fma_meta_to_csv()

    # Convert stuff to npy files
    # mp3_to_npy(
    #     'ambient_electro',  # Which genre
    #     Path('data/fma_large.zip'),  # Path to zip file
    #     Path('data/npy_data'),  # Where to save spectrograms
    #     Path('data/processed_data')  # Where to save paths to spectrograms
    # )
