import os
import cv2
import matplotlib.pyplot as plt

def visualize_label(label, data_dir="data", n_samples=30):
    label_dir = os.path.join(data_dir, f"label_{label}")
    if not os.path.exists(label_dir):
        print(f"Không tồn tại {label_dir}")
        return
    files = [f for f in os.listdir(label_dir) if f.endswith(".png")][:n_samples]
    plt.figure(figsize=(15, 2))
    for idx, file in enumerate(files):
        img = cv2.imread(os.path.join(label_dir, file), cv2.IMREAD_GRAYSCALE)
        plt.subplot(1, n_samples, idx+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(file, fontsize=6)
    plt.suptitle(f"label_{label}")
    plt.show()

if __name__ == "__main__":
    visualize_label(9)
