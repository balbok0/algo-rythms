from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

class SpectrogramSequenceDataset(Dataset):
    def __init__(self, paths: np.ndarray, sequence_length: int, batch_size: int, shuffle: bool = True, transform=None):
        super().__init__()
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        paths = np.random.permutation(paths) if shuffle else paths

        # Make an array of tuples containing
        # spectrogram paths,and indexes to start sequences at in those spectrograms
        file_idxs = []
        for path in paths:
            spectrogram = np.load(path)

            for idx in range(0, spectrogram.shape[1], sequence_length):
                file_idxs.append((path, idx))


        # Make file_idxs a multiple of batch_size, so it's reshapeable
        file_idxs = file_idxs[:len(file_idxs) - len(file_idxs) % batch_size]

        self.file_idxs = np.reshape(file_idxs, (batch_size, -1))

    def __len__(self):
        return np.prod(self.file_idxs.shape)

    def __getitem__(self, idx):
        i = idx // self.num_batches
        j = idx % self.num_batches
        path, start_idx = self.file_idxs[i, j]
        spectrogram = np.load(path)

        data = spectrogram[:, start_idx:start_idx + self.sequence_length]
        target = spectrogram[:, start_idx + 1:start_idx + self.sequence_length + 1]

        # Data can be pulled exactly from the end of the file.
        # In that case we need to pad zeros (silence) to the end of target
        if target.shape[1] < self.sequence_length:
            target = np.pad(target, [(0, 0), (0, self.sequence_length - target.shape[1])])

        if self.transform:
            data = self.transform(data)
            target = self.transform(target)

        return data, target