import os
import glob
import numpy as np
import librosa
from sklearn import preprocessing
import cadnn
from record import Recording


model = cadnn.build_model(5, 2)
model.load_weights('model/model.h5')

record = Recording()
os.system('clear')
print('*** ENTERを押して録音開始・終了 ***')

#files = glob.glob('data/context1/*/*.wav')
#classes = list(set([f.split('/')[-2] for f in files]))
classes = ['a', 'i', 'u', 'e', 'o']

mode = 0  # 0：録音開始，1：録音終了
cnt = 1

while True:
    key = input()

    if mode == 0:
        # 録音開始
        print('===== {0} START ==============='.format(cnt))
        record.record_start.set()
        record.record_end.clear()
        mode = 1

    else:
        # 録音終了
        print('===== END ===============')
        record.record_start.clear()
        while not record.record_end.is_set():
            pass
        mode = 0
        cnt += 1

        x1 = []
        x2 = []
        wav, fs = librosa.load('tmp/voice.wav', sr=8000)
        context_feature = librosa.feature.mfcc(wav, sr=fs, hop_length=10**6, htk=True).T[0]
        mfcc = librosa.feature.mfcc(wav, sr=fs, n_mfcc=32).T
        for frame in mfcc:
            x1.append(preprocessing.minmax_scale(frame))
            x2.append(context_feature)

        pred = [classes[np.argmax(p)] for p in model.predict([x1, x2])]
        print(pred)

