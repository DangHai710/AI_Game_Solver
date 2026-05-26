import os
import shutil
import cv2

def label_and_move_cells(cells_dir="output_cells", data_dir="data"):
    cell_files = sorted([f for f in os.listdir(cells_dir) if f.endswith(".png")])
    for cell_file in cell_files:
        cell_path = os.path.join(cells_dir, cell_file)
        img = cv2.imread(cell_path)
        cv2.imshow("Cell", img)
        cv2.waitKey(1)  # Hiển thị ảnh
        label = input(f"Nhập nhãn cho {cell_file} (0-9, bỏ qua: Enter): ").strip()
        cv2.destroyAllWindows()
        if label == "":
            print(f"Bỏ qua {cell_file}")
            continue
        if not label.isdigit() or not (0 <= int(label) <= 9):
            print("Nhãn không hợp lệ. Bỏ qua.")
            continue
        dest_dir = os.path.join(data_dir, f"label_{label}")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        dest_path = os.path.join(dest_dir, cell_file)
        shutil.move(cell_path, dest_path)
        print(f"Đã chuyển {cell_file} vào {dest_dir}")

if __name__ == "__main__":
    label_and_move_cells()
