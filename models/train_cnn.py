import glob
import os

import cv2
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def load_data():
    x, y = [], []
    for label in range(10):
        for path in glob.glob(f"data/label_{label}/*.png"):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            x.append(img.astype("float32") / 255.0)
            y.append(label)
    return np.array(x).reshape(-1, 28, 28, 1), to_categorical(y, 10)


def build_model():
    return Sequential(
        [
            Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            BatchNormalization(),
            Conv2D(32, 3, activation="relu"),
            MaxPooling2D(),
            Dropout(0.25),
            Conv2D(64, 3, activation="relu"),
            BatchNormalization(),
            Conv2D(64, 3, activation="relu"),
            MaxPooling2D(),
            Dropout(0.25),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )


if __name__ == "__main__":
    X, y = load_data()
    if len(X) == 0:
        raise SystemExit("Chua co du lieu trong data/label_0..9")

    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.08, height_shift_range=0.08, zoom_range=0.08)
    model = build_model()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(datagen.flow(X, y, batch_size=32), epochs=40, validation_data=(X, y))

    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_sudoku.h5")
    print("Da luu model vao models/cnn_sudoku.h5")
