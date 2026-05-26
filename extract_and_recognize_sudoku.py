import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from utils.vision import preprocess_cell  # Giả sử bạn có hàm này để xử lý ô

def select_image():
    path = input("Nhập đường dẫn ảnh Sudoku tổng thể: ").strip()
    if not os.path.isfile(path):
        print("Ảnh không tồn tại!")
        return None
    return path

def find_sudoku_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def crop_cells(grid_img, out_dir="output_cells"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    h, w = grid_img.shape[:2]
    cell_h, cell_w = h // 9, w // 9
    cell_paths = []
    for i in range(9):
        for j in range(9):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = grid_img[y1:y2, x1:x2]
            cell_path = os.path.join(out_dir, f"cell_{i}_{j}.png")
            cv2.imwrite(cell_path, cell)
            cell_paths.append(cell_path)
    return cell_paths

def recognize_cells(cell_paths, model_path="models/cnn_sudoku.h5"):
    model = load_model(model_path)
    results = []
    # Tạo thư mục lưu ảnh đã preprocess
    out_dir = "output_cells_preprocessed"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for path in cell_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cell_img = preprocess_cell(img)
        # Lưu ảnh đã preprocess để kiểm tra
        save_img = (cell_img.squeeze() * 255).astype('uint8')
        base = os.path.basename(path)
        cv2.imwrite(os.path.join(out_dir, base), save_img)
        pred = model.predict(np.expand_dims(cell_img, axis=0))
        label = np.argmax(pred)
        results.append(label)
        print(f"{path}: {label}")
    print(f"Đã lưu ảnh đã preprocess vào {out_dir}/ để kiểm tra chất lượng.")
    return results

def main():
    img_path = select_image()
    if not img_path:
        return
    img = cv2.imread(img_path)
    grid = find_sudoku_grid(img)
    if grid is None:
        print("Không tìm thấy khung Sudoku!")
        return
    warped = four_point_transform(img, grid)
    cell_paths = crop_cells(warped)
    print(f"Đã cắt {len(cell_paths)} ô nhỏ.")
    # Nhận diện số trong từng ô
    recognize = input("Bạn có muốn nhận diện số trong từng ô luôn không? (y/n): ").strip().lower()
    if recognize == 'y':
        recognize_cells(cell_paths)
    else:
        print("Đã lưu các ô nhỏ, bạn có thể gán nhãn sau.")

if __name__ == "__main__":
    main()
