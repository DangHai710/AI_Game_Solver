import os
import socket

import cv2
import numpy as np
import streamlit as st

from sudoku import solve_sudoku
from utils.vision import bgr_to_rgb, find_grid_contour, overlay_solution_on_original, split_cells, warp_grid

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None


MODEL_PATH = "models/cnn_sudoku.h5"
OUTPUT_DIR = "output_cells/latest"


@st.cache_resource(show_spinner=False)
def load_digit_model():
    return load_model(MODEL_PATH, compile=False) if load_model and os.path.exists(MODEL_PATH) else None


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "IP-may-tinh"


def decode_image(file):
    if file is None:
        return None
    data = np.frombuffer(file.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_cells(cells):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    arr = np.array([[cells[r][c]["display"] for c in range(9)] for r in range(9)], dtype=np.uint8)
    np.save("cell_imgs.npy", arr)
    for r in range(9):
        for c in range(9):
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"cell_{r}_{c}.png"), arr[r][c])


def recognize(cells, model, threshold):
    board = np.zeros((9, 9), dtype=np.int32)
    conf = np.zeros((9, 9), dtype=np.float32)
    batch, pos = [], []

    for r in range(9):
        for c in range(9):
            cell = cells[r][c]
            if cell["is_empty"] or (cell["ink_ratio"] < 0.02 and cell.get("component_count", 0) >= 2):
                continue
            batch.append(cell["model"])
            pos.append((r, c))

    if batch:
        preds = model.predict(np.array(batch), verbose=0)
        for (r, c), pred in zip(pos, preds):
            digit, score = int(np.argmax(pred)), float(np.max(pred))
            conf[r, c] = score
            if digit and score >= threshold:
                board[r, c] = digit
    return board, conf


def board_image(board):
    img = np.full((450, 450, 3), 255, np.uint8)
    for i in range(10):
        thick = 3 if i % 3 == 0 else 1
        cv2.line(img, (0, i * 50), (450, i * 50), (210, 210, 210), thick)
        cv2.line(img, (i * 50, 0), (i * 50, 450), (210, 210, 210), thick)
    for r in range(9):
        for c in range(9):
            if board[r][c]:
                cv2.putText(img, str(int(board[r][c])), (c * 50 + 15, r * 50 + 37), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2)
    return img


st.set_page_config(page_title="AI Game Solver - Sudoku", layout="wide")
st.title("AI Game Solver: Sudoku")

with st.sidebar:
    st.header("Chay qua LAN")
    st.code("streamlit run app.py --server.address 0.0.0.0 --server.port 8501")
    st.caption(f"Dien thoai cung Wi-Fi/LAN mo: http://{lan_ip()}:8501")
    threshold = st.slider("Nguong tin cay", 0.1, 0.95, 0.75, 0.05)

upload_tab, camera_tab = st.tabs(["Chon anh de upload", "Camera truc tiep"])
with upload_tab:
    image = decode_image(st.file_uploader("Chon anh tu may tinh/dien thoai", type=["jpg", "jpeg", "png"]))
with camera_tab:
    cam = decode_image(st.camera_input("Chup truc tiep neu trinh duyet cho phep camera"))
    image = cam if cam is not None else image

if image is None:
    st.info("Hay upload anh Sudoku hoac chup truc tiep bang camera.")
    st.stop()

model = load_digit_model()
if model is None:
    st.warning("Chua thay models/cnn_sudoku.h5. Ban van cat duoc 81 o, nhung can model de giai Sudoku.")
contour = find_grid_contour(image)
if contour is None:
    st.error("Khong tim duoc khung Sudoku. Hay crop sat bang hon hoac chup thang goc hon.")
    st.image(bgr_to_rgb(image), caption="Anh goc", use_container_width=True)
    st.stop()

warped, _ = warp_grid(image, contour)
cells = split_cells(warped)
save_cells(cells)

st.subheader("Anh goc")
st.image(bgr_to_rgb(image), use_container_width=True)

st.subheader("81 o da cat")
for r in range(9):
    cols = st.columns(9)
    for c in range(9):
        cols[c].image(cells[r][c]["display"], width=52)
st.caption("Da luu cell_imgs.npy va output_cells/latest/cell_r_c.png de phuc vu gan nhan/train.")

if st.button("Giai Sudoku", type="primary", disabled=model is None):
    board, confidence = recognize(cells, model, threshold)
    solved = board.copy()
    ok = solve_sudoku(solved)

    left, right = st.columns(2)
    with left:
        st.subheader("Cac so da nhan dien")
        st.image(board_image(board), use_container_width=True)
    with right:
        st.subheader("Anh goc da dien so con thieu")
        if ok:
            st.image(bgr_to_rgb(overlay_solution_on_original(image, contour, board, solved)), use_container_width=True)
        else:
            st.error("Board nhan dien dang mau thuan, can tang nguong tin cay hoac train lai data.")

    with st.expander("Do tin cay tung o"):
        st.dataframe(np.round(confidence, 3), use_container_width=True)