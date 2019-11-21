from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

class SpectrogramImageDataset(Dataset):
    def __init__(self, paths: np.ndarray, y_size: int, transform=None):
        """Loads Spectrograms as Images. Can be used for Convolution GANs etc.

        Arguments:
            paths {np.ndarray} -- Array of strings representing paths to spectrograms.
            y_size {int} -- Spectrograms can have different duration in time.
                Make sure, that Conv nets are ok to use, by reshaping time axis to this size.

        Keyword Arguments:
            transform {[type]} -- PyTorch transforms to apply to data (ToTensor should be there) (default: {None})
        """
        super().__init__()
        self.paths = paths
        self.y_size = y_size
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample = np.load(self.paths[idx])

        # Decrease size, or pad with zeros (i.e. silence)
        if sample.shape[1] > self.y_size:
            sample = sample[:self.y_size]
        else:
            sample = np.pad(
                sample,
                [(0, 0), (0, self.y_size - sample.shape[1])]
            )
        # Add grayscale channel
        sample = sample[np.newaxis, ...]

        if self.transform:
            sample = self.transform(sample)

        return sample