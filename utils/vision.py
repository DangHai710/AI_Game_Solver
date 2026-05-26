import cv2
import numpy as np

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    diff = np.diff(myPoints, axis=1)
    # 0: trên trái (min add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    # 1: trên phải (min diff)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    # 2: dưới phải (max add)
    myPointsNew[2] = myPoints[np.argmax(add)]
    # 3: dưới trái (max diff)
    myPointsNew[3] = myPoints[np.argmax(diff)]
    return myPointsNew

def get_grid_contour(contours):
    grid_con = np.array([])
    max_area = 0
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i in contours:
        area = cv2.contourArea(i)
        if area < 10000:
            continue
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        if area > max_area and len(approx) == 4 and cv2.isContourConvex(approx):
            grid_con = approx
            max_area = area
            break
    return grid_con

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    thresh_inv = cv2.bitwise_not(thresh)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, kernel)
    kernel2 = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel2)
    return closed

def warp_perspective(img, grid_con, size=450):
    # Nếu tìm được contour 4 đỉnh, căn chỉnh lại khung Sudoku về hình vuông
    if grid_con.shape[0] == 4:
        pts = grid_con.reshape(4, 2)
        pts = reorder(pts)
        pts1 = np.float32(pts)
        pts2 = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, matrix, (size, size))
        return img_warp
    return img

# Tiền xử lý ô số cho CNN
def preprocess_cell(cell_img, size=28):
    # Nếu ảnh là 3 kênh thì chuyển về xám
    if len(cell_img.shape) == 3:
        cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    # Resize về size x size
    cell_img = cv2.resize(cell_img, (size, size))
    # Chuẩn hóa về [0, 1]
    cell_img = cell_img.astype('float32') / 255.0
    # Thêm chiều kênh nếu cần (cho CNN)
    cell_img = np.expand_dims(cell_img, axis=-1)
    return cell_img