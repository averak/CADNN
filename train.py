#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import librosa
from sklearn import preprocessing
import glob
import tqdm
import cadnn


def feature_extract():
    #--------------------------------------------
    # .wavファイルを読み込み
    #--------------------------------------------

    files = glob.glob('data/context1/*/*.wav')
    classes = list(set([f.split('/')[-2] for f in files]))

    x = []
    y = []

    for f in files:
        wav, fs = librosa.load(f, sr=8000)
        mfcc = librosa.feature.mfcc(wav, sr=fs, n_mfcc=32).T

        for frame in mfcc:
            x.append(preprocessing.minmax_scale(frame))
            y.append(classes.index(f.split('/')[-2]))

    x = np.array(x)
    y = np.array(y)

    # ランダムに並べ替え
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    return x, y


# 特徴量抽出
x, y = feature_extract()

# クラス -> [a, i, u, e, o]
# コンテキスト -> [man, woman]
model = cadnn.build_model(5, 2)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"],
)
# save model image
tf.keras.utils.plot_model(model, to_file='model/architecture.png')


model.fit(
    [x, x],
    y,
    batch_size=32,
    epochs=50,
)

