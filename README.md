## AI Game Solver - Sudoku

Ứng dụng Streamlit dùng OpenCV và CNN để quét ảnh Sudoku, cắt 81 ô, nhận diện chữ số và giải bằng Backtracking + MRV.

## Cấu trúc

```text
AI_Game_Solver/
├─ app.py
├─ sudoku.py
├─ utils/
│  └─ vision.py
├─ data/
│  ├─ label_0/
│  └─ ...
├─ models/
│  ├─ train_cnn.py
│  └─ cnn_sudoku.h5
├─ save_cells.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy web app

Chạy trên máy tính:

```bash
streamlit run app.py
```

Cho điện thoại cùng Wi‑Fi/LAN truy cập:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Mở địa chỉ LAN hiển thị trong sidebar, ví dụ:

```text
http://192.168.x.x:8501
```

## Hướng sử dụng

1. Tab "Chọn ảnh để upload": chọn ảnh từ máy tính hoặc điện thoại. Trình duyệt trên điện thoại cho phép chụp mới hoặc lấy ảnh từ thư viện.
2. Tab "Camera trực tiếp": dùng camera nếu trình duyệt cho phép. Trên điện thoại có thể bị chặn nếu app chạy qua HTTP (không HTTPS).
3. Sau khi có ảnh, app hiển thị "Ảnh gốc" và "81 ô đã cắt".
4. App tự động lưu 81 ô vào `cell_imgs.npy` và `output_cells/latest/cell_r_c.png`.
5. Bấm "Giai Sudoku" để hiển thị bảng số đã nhận diện và ảnh gốc đã điền số còn thiếu.
6. Mở "Độ tin cậy từng ô" để xem confidence phục vụ báo cáo và debug.

## Xử lý ảnh

- `utils/vision.py` tìm contour Sudoku bằng threshold thích nghi và chọn contour lớn nhất có 4 đỉnh.
- Lưới được biến đổi về ảnh vuông `450x450`.
- Mỗi ô được cắt theo lưới 9x9, sau đó lấy 80% vùng trung tâm để tránh viền đường.
- Ảnh ô được threshold, lọc connected components để loại nhiễu, resize về `28x28` cho CNN.

## Gán nhãn dữ liệu

Sau khi app tạo `cell_imgs.npy`, chạy:

```bash
python save_cells.py
```

Bấm phím `0-9` để lưu ô hiện tại vào `data/label_x`, bấm `Space` để bỏ qua, `ESC` để dừng.

## Huấn luyện CNN

```bash
python models/train_cnn.py
```

Model sẽ được lưu vào:

```text
models/cnn_sudoku.h5
```

## File chính

- `sudoku.py`: Backtracking + MRV.
- `app.py`: giao diện Streamlit, upload/camera, nhận diện và hiển thị kết quả.
- `utils/vision.py`: tìm khung, xử lý ảnh, cắt 81 ô, tiền xử lý ô, vẽ lời giải lên ảnh gốc.
- `save_cells.py`: gán nhãn 81 ô đã cắt.
- `models/train_cnn.py`: huấn luyện CNN từ `data/label_0..9`.
