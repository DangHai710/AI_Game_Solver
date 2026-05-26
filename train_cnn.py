
import os
import glob
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load dữ liệu và augmentation cho các nhãn yếu
X, y = [], []
aug_labels = list(range(10))
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)
for label in range(10):
    files = glob.glob(f'data/label_{label}/*.png')
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = img / 255.0
        X.append(img)
        y.append(label)
        # Augmentation: mỗi số 50 lần, riêng số 9 là 80 lần
        if label in aug_labels:
            img4d = img.reshape(1, 28, 28, 1)
            aug_iter = datagen.flow(img4d, batch_size=1)
        #  /* for _ in range(10)// Chỉ augment 10 lần cho mỗi ảnh gốc để tránh quá tải dữ liệu */
            n_aug = 80 if label == 9 else 50
            for _ in range(n_aug):
                aug_img = next(aug_iter)[0].reshape(28, 28)
                X.append(aug_img)
                y.append(label)
X = np.array(X).reshape(-1, 28, 28, 1)
y = to_categorical(y, 10)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=40, batch_size=32, validation_split=0.1)
model.save('models/cnn_sudoku.h5')
print('Đã lưu model vào models/cnn_sudoku.h5')
