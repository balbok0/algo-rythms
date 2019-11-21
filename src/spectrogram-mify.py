import librosa as lbs
import librosa.display as lbs_dsp
import matplotlib.pyplot as plt
import numpy as np
import time
import glob

def spectrogrammify():
    y, sr = lbs.load("data/testing_data/testfile.mp3")
    duration = lbs.core.get_duration(y, sr)
    y = y[:int(len(y) * 30 / duration)]

    plt.figure(figsize=(12, 8))
    D = lbs.amplitude_to_db(np.abs(lbs.stft(y)), ref=np.max)

    np.save('data/testing_data/testfile.npy', D)
    np.savez_compressed('data/testing_data/testfile.npz', **{"0": D})

    start = time.time()
    compressed = np.load('data/testing_data/testfile.npz')
    print('Time in ms: {}'.format(time.time() - start))

    plt.subplot(4, 2, 1)
    lbs_dsp.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.show()


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
