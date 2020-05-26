#!/usr/bin/env python
import cadnn


# クラス -> [a, i, u, e, o]
# コンテキスト -> [man, woman]
model = cadnn.build_model(5, 2)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"],
)

