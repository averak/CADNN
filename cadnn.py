from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Multiply, Add


def build_model(n_class, n_context):
    #--------------------------------------------
    # モデルをビルド
    # params:
    #   - n_class:int -> 分類するクラス数
    #   - n_context:int -> コンテキスト数
    # return:
    #   - tensorflow.python.keras.engine.training.Model
    #--------------------------------------------

    # 入力層
    main_input = Input(shape=(32,))
    sub_input = Input(shape=(20,))

    # サブネットワーク
    sub_x = Dense(32, activation='relu')(sub_input)
    sub_y1 = Dense(1, activation='relu')(sub_x)
    sub_y2 = Dense(1, activation='relu')(sub_x)

    # メインネットワーク
    main_x1 = Dense(32, activation='relu')(main_input)
    main_x2_1 = Dense(32, activation='relu')(main_x1)
    main_x2_1 = Multiply()([main_x2_1, sub_y1])
    main_x2_2 = Dense(32, activation='relu')(main_x1)
    main_x2_2 = Multiply()([main_x2_2, sub_y2])
    main_x2 = Add()([main_x2_1, main_x2_2])
    main_x3 = Dense(64, activation='relu')(main_x2)
    main_x4 = Dense(64, activation='relu')(main_x3)
    main_y = Dense(n_class, activation='softmax')(main_x4)

    model = Model(inputs=[main_input, sub_input], outputs=main_y)
    return model

