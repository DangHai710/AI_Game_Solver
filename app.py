import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from utils.vision import preprocess_image, get_grid_contour, reorder
from utils.helpers import to_ping
from engines.sudoku import solve_sudoku


st.set_page_config(page_title="AI Game Solver", layout="wide")
st.title("🧩 AI Game Solver: Sudoku")

mode = st.sidebar.radio("Trò chơi:", ["Sudoku"])

# Load model
model_path = 'models/cnn_sudoku.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error("Chưa thấy file models/cnn_sudoku.h5! Hãy chạy train_cnn.py để huấn luyện model mới.")

img_file = st.camera_input("Quét bảng")

if img_file:
    file_bytes = np.frombuffer(img_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Tiền xử lý: chỉ lấy 1 ảnh trắng đen duy nhất
    bw_img = preprocess_image(img)
    st.image(bw_img, caption="1. Ảnh trắng đen sau tiền xử lý")

    # Tìm contour trên ảnh trắng đen
    contours, _ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grid_con = get_grid_contour(contours)

    if grid_con.size != 0:
        # Vẽ contour lên ảnh gốc
        img_contour = img.copy()
        cv2.drawContours(img_contour, [grid_con], -1, (0,255,0), 3)
        st.image(img_contour, caption="2. Ảnh gốc với contour khung Sudoku")

        grid_con_ordered = reorder(grid_con)
        pts1 = np.float32(grid_con_ordered)
        pts2 = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # Nắn thẳng trên ảnh gốc
        warped_color = cv2.warpPerspective(img, M, (450, 450))
        # Tiền xử lý lại trên ảnh nắn thẳng
        warped = preprocess_image(warped_color)
        st.image(warped, caption="3. Ảnh Sudoku đã nắn thẳng (trắng đen)")

        # Cắt ô số ngay sau khi nắn thẳng
        size = 9 if mode == "Sudoku" else 6
        side = 450 // size
        cell_imgs = []
        for i in range(size):
            row_imgs = []
            for j in range(size):
                cell = warped[i*side:(i+1)*side, j*side:(j+1)*side]
                cell = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)
                _, cell_bin = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cell_bin = cv2.bitwise_not(cell_bin)
                row_imgs.append(cell_bin)
            cell_imgs.append(row_imgs)
        # Hiển thị từng ô số nhỏ
        st.write("4. Các ô số sau khi cắt ra:")
        for i in range(size):
            cols = st.columns(size)
            for j in range(size):
                cols[j].image(cell_imgs[i][j], width=40)

        # Nút giải sudoku chỉ nhận diện số và giải, không cắt lại ảnh
        if st.button(f"Giải {mode}"):
            res = []
            for i in range(size):
                for j in range(size):
                    cell_bin = cell_imgs[i][j]
                    feat = cell_bin.reshape(1, 28, 28, 1) / 255.0
                    pred = model.predict(feat)
                    res.append(int(np.argmax(pred)))
            board = np.array(res).reshape(size, size)
            st.write("Dữ liệu nhận diện:")
            st.write(board)
            # Lưu cell_imgs ra file để gán nhãn và huấn luyện sau này
            np.save('cell_imgs.npy', cell_imgs)
            st.success('Đã lưu cell_imgs.npy. Bạn có thể chạy save_cells.py để gán nhãn và lưu ảnh.')
            if mode == "Sudoku":
                if solve_sudoku(board):
                    st.success("Lời giải:")
                    st.table(board)
    else:
        st.warning("Không nhận diện được khung Sudoku. Hãy đảm bảo bảng vuông góc, đủ sáng và chiếm phần lớn khung hình!")