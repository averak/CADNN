from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout


def build_model(n_class, n_context):
    #--------------------------------------------
    # モデルをビルド
    # params:
    #   - n_class:int -> 分類するクラス数
    #   - n_context:int -> コンテキスト数
    # return:
    #   - tensorflow.python.keras.engine.training.Model
    #--------------------------------------------

    # メインネットワークの入力層
    main_input = Input(shape=(32,))
    # サブネットワークの入力層
    sub_input = Input(shape=(32,))

    # メインネットワークの中間層
    main_x1 = Dense(64, activation='relu')(main_input)
    main_x2 = Dense(64, activation='relu')(main_x1)
    main_x3 = Dense(64, activation='relu')(main_x2)

    # 出力層
    y = Dense(n_class, activation='softmax')(main_x3)

    # モデル
    model = Model(inputs=main_input, outputs=y)
    return model

