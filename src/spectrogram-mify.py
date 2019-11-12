import librosa as lbs
import matplotlib as plt
import numpy as np
import soundfile as sf


def spectrogrammify():
    y, sr = lbs.load("000900.mp3")
    plt.figure(figsize=(12, 8))
    D = lbs.amplitude_to_db(np.abs(lbs.stft(y)), ref=np.max)
    plt.subplot(4, 2, 1)
    lbs.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.show()


if __name__ == "__main__":
    spectrogrammify()
