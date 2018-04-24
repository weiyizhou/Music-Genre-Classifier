import librosa
import sklearn as skl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa.display
import os
import shutil
import math
from PIL import Image
import eyed3
from subprocess import Popen, PIPE, STDOUT



def main():
    tracks = loadMetadata()
    sliceSpect(tracks)

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
    counter = 0
    # Create spectograms
    for filepath in mp3Files:
        temp = eyed3.load(filepath)
        if temp == None:
            print(counter)
            continue
        if temp.info.mode =='Mono':
            command = "cp '{}' '{}.mp3'".format(filepath, 'temp')
        else:
            command = "sox '{}' '{}.mp3' remix 1,2".format(filepath, 'temp')
        p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=os.getcwd())
        output, errors = p.communicate()
        if errors:
            print(errors)
        destination = "spectograms/" + os.path.splitext(os.path.basename(filepath))[0]
        command = "sox '{}.mp3' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format('temp', 50, destination)
        p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=os.getcwd())
        output, errors = p.communicate()
        if errors:
            print(errors)
        else:
            counter += 1
            #print(counter)
        # Remove tmp mono track
        if os.path.exists("{}.mp3".format('temp')):
            os.remove("{}.mp3".format('temp'))
    print(counter)

def loadMetadata():
    genres = ['Rock', 'Electronic', 'Experimental', 'Hip-Hop', 'Folk', 'Instrumental', 'Pop', 'International']
    tracks = pd.read_csv("fma_metadata/tracks.csv", sep=',', quotechar='"', header=1, mangle_dupe_cols=True)
    tracks = tracks[['Unnamed: 0', 'title', 'genres', 'genre_top', 'split', 'subset']]
    tracks = tracks[tracks.subset == "small"]
    tracks['genre_top'] = tracks['genre_top'].apply(lambda x: genres.index(x))
    tracks['Unnamed: 0'] = tracks['Unnamed: 0'].apply(lambda x: int(x))
    tracks.drop(columns=['title', 'genres', 'split', 'subset'], inplace=True)
    tracks = tracks.as_matrix()
    mapping = {}
    for each in tracks:
        mapping[each[0]] = each[1]
    return mapping

def sliceSpect(tracks):
    if os.path.exists("slices"):
        shutil.rmtree("slices")
    os.makedirs("slices")
    for filename in os.listdir(os.getcwd() + "/spectograms"):
        orgImg = Image.open("spectograms/" + filename)
        #print(orgImg.size)
        w, h = orgImg.size
        sliceCount = int(math.floor(w/128))
        for x in range(sliceCount):
            sliceImg = orgImg.crop((x*128, 0, x*128 + 128, 128))
            #print(sliceImg.size)
            sliceImg.save("slices/" + os.path.splitext(filename)[0] + "_" + str(x) + ".png")


def loadSlices(tracks):
    for filename in os.listdir(os.getcwd() + "/slices"):
        img = Image.open(filename)
        imageSize = 128
        img = img.resize((imageSize, imageSize), resample=Image.ANTIALIAS)
        imgData = np.asarray(img, dtype=np.uint8).reshape(imageSize, imageSize, 1)
        imgData = imgData / 255.
        return imgData








if __name__ == "__main__":
    main()
