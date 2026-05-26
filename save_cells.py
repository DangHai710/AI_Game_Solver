import cv2
import os

def save_cells(cell_imgs):
    for i, row in enumerate(cell_imgs):
        for j, cell in enumerate(row):
            cv2.imshow('Cell', cell)
            key = cv2.waitKey(0)
            if ord('0') <= key <= ord('9'):
                label = chr(key)
                save_dir = f'data/label_{label}'
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(f'{save_dir}/cell_{i}_{j}.png', cell)
            elif key == 27:  # ESC để thoát
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()

# Ví dụ sử dụng:
# cell_imgs = ... # List 2 chiều các ô số 28x28
# save_cells(cell_imgs)
import numpy as np
from save_cells import save_cells
cell_imgs = np.load('cell_imgs.npy', allow_pickle=True)
save_cells(cell_imgs)