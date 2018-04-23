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
import os
import shutil


def main():
    #createSpectograms()
    loadMetadata()


def createSpectograms():
    # Get all mp3Files in fma_small (must be in same dir as this python file)
    mp3Files = []
    for root, dirs, files in os.walk(os.getcwd() + "/fma_small", topdown=False):
        for name in files:
            if name.lower().endswith(".mp3"):
                mp3Files.append(os.path.join(root, name))
    print(len(mp3Files))
    if os.path.exists("spectograms"):
        shutil.rmtree("spectograms")
    os.makedirs("spectograms")

    # Create spectograms
    i = 0 #REMOVE AFTER TEST
    for filepath in mp3Files:
        x, sr = librosa.load(filepath, sr=None, mono=True)
        print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
        log_mel = librosa.core.power_to_db(mel)
        librosa.display.specshow(log_mel, sr=sr, hop_length=512)
        plt.axis('off')
        plt.savefig("spectograms/" + os.path.splitext(os.path.basename(filepath))[0] + ".png", bbox_inches='tight',
                    pad_inches=-0.1)
        #for testing only
        if (i > 2):
            break
        i += 1


def loadMetadata():
    genres = ['Rock', 'Electronic', 'Experimental', 'Hip-Hop', 'Folk', 'Instrumental', 'Pop', 'International']
    tracks = pd.read_csv("fma_metadata/tracks.csv", sep=',', quotechar='"', header=1, usecols=['id', 'title', 'genres', 'genre_top', 'split', 'subset'])
    tracks = tracks[tracks.subset == "small"]
    tracks['genre_top'] = tracks['genre_top'].apply(lambda x: genres.index(x))
    tracks.drop(columns=['title', 'genres', 'split', 'subset'], inplace=True)
    tracks.id.astype(int)
    tracks.genre_top.astype(int)
    tracks = tracks.as_matrix()
    return tracks



if __name__ == "__main__":
    main()

