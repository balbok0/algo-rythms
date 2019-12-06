import librosa as lbs
import librosa.display as lbs_dsp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import tqdm


# Given a folder with the predicted outputs, it generates and saves the spectrogram
# for each file. Files should be named by number. Total outputs will default to 100.
def prepare_output_graphs(predicted_outputs: Path, total_outputs: int = 100):
    for i in tqdm.trange(total_outputs):
        file_audio = Path('{}.wav'.format(i))
        file_spec = Path('{}.png'.format(i))
        y, sr = lbs.load(predicted_outputs / file_audio)
        duration = lbs.core.get_duration(y, sr)
        y = y[:int(len(y) * 30 / duration)]

        plt.figure(figsize=(12, 8))
        D = lbs.amplitude_to_db(np.abs(lbs.stft(y)), ref=np.max)

        lbs_dsp.specshow(D, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Linear-frequency power spectrogram')
        plt.savefig(predicted_outputs / file_spec)


def show_spectro():
    visited_lengths = []
    for p in glob.glob('data/npy_data/**/*.npy'):
        D = np.load(p)
        if D.shape[1] not in visited_lengths:
            visited_lengths.append(D.shape[1])

            plt.figure(figsize=(12, 8))

            lbs_dsp.specshow(D, y_axis='linear')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Linear-frequency power spectrogram')
            plt.show()


            plt.imshow(D)
            plt.title('Vanilla D (Make sure color ain\'t logged)')
            plt.show()

if __name__ == "__main__":
    # spectrogrammify()
    show_spectro()
