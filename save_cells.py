import os
import time

import cv2
import numpy as np


def save_cell(img, label, row, col):
    out_dir = f"data/label_{label}"
    os.makedirs(out_dir, exist_ok=True)
    name = f"cell_{row}_{col}_{int(time.time() * 1000)}.png"
    cv2.imwrite(os.path.join(out_dir, name), img)


if __name__ == "__main__":
    cells = np.load("cell_imgs.npy", allow_pickle=True)
    print("Bam 0-9 de gan nhan, Space de bo qua, ESC de dung.")

    for r in range(9):
        for c in range(9):
            img = cells[r][c]
            preview = cv2.resize(img, (160, 160), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(f"cell {r},{c}", preview)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key == 27:
                raise SystemExit
            if ord("0") <= key <= ord("9"):
                save_cell(img, chr(key), r, c)
