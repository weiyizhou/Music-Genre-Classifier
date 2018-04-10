import librosa
import sklearn as skl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa.display
import utils
import keras


def main():
    x, sr = librosa.load("fma_small/000/000002.mp3", sr=None, mono=True)
    print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    log_mel = librosa.core.power_to_db(mel)
    librosa.display.specshow(log_mel, sr=sr, hop_length=512)
    plt.axis('off')
    plt.savefig("test.png", bbox_inches='tight', pad_inches=-0.1)


if __name__ == "__main__":
    main()

