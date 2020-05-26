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

    x1 = []
    x2 = []
    y = []

    for f in files:
        # 音声読み込み
        wav, fs = librosa.load(f, sr=8000)
        # mfccをコンテキストの特徴量とする
        context_feature = librosa.feature.mfcc(wav, sr=fs, hop_length=10**8, htk=True)

        mfcc = librosa.feature.mfcc(wav, sr=fs, n_mfcc=32).T
        for frame in mfcc:
            x1.append(preprocessing.minmax_scale(frame))
            x2.append(context_feature.reshape((20)))
            y.append(classes.index(f.split('/')[-2]))

    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    # ランダムに並べ替え
    perm = np.arange(len(x1))
    np.random.shuffle(perm)
    x1 = x1[perm]
    x2 = x2[perm]
    y = y[perm]

    return x1, x2, y


# 特徴量抽出
x1, x2, y = feature_extract()

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
    [x1, x2],
    y,
    batch_size=32,
    epochs=200,
)

