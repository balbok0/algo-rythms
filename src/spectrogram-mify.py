import librosa as lbs
import librosa.display as lbs_dsp
import matplotlib.pyplot as plt
import numpy as np
import time

def spectrogrammify():
    # y, sr = lbs.load("data/testing_data/testfile.mp3")
    # plt.figure(figsize=(12, 8))
    # D = lbs.amplitude_to_db(np.abs(lbs.stft(y)), ref=np.max)
    # print('y: {}'.format(y.shape))
    # print('D: {}'.format(D.shape))
    # np.savez_compressed('data/testing_data/testfile.npz', **{'0': D})

    start = time.time()
    compressed = np.load('data/testing_data/testfile.npz')
    print(compressed['0'].shape)
    print('Time in ms: {}'.format(time.time() - start))
    exit(0)

    plt.subplot(4, 2, 1)
    lbs_dsp.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.show()


if __name__ == "__main__":
    spectrogrammify()
