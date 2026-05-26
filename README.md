# AI Game Solver

Ứng dụng nhận diện và giải Sudoku bằng Computer Vision + CNN + Streamlit.

## Tính năng

- Quét bảng Sudoku từ camera
- Tiền xử lý ảnh, tìm khung bảng và tách từng ô
- Nhận diện chữ số bằng mô hình CNN
- Giải Sudoku sau khi nhận diện xong
- Hỗ trợ lưu ảnh ô số để gán nhãn và huấn luyện lại

## Cài đặt

### 1. Tạo môi trường ảo

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 2. Cài thư viện

```bash
pip install -r requirements.txt
```

Nếu máy bạn chưa có TensorFlow, cài thêm:

```bash
pip install tensorflow
```

## Chạy ứng dụng

Sau khi có model trong `models/cnn_sudoku.h5`, chạy:

```bash
streamlit run app.py
```

## Huấn luyện lại model

Nếu muốn train lại mô hình nhận diện số, chạy:

```bash
python train_cnn.py
```

Script này sẽ đọc dữ liệu trong `data/`, tăng cường ảnh, huấn luyện CNN và lưu model vào `models/cnn_sudoku.h5`.

## Gán nhãn dữ liệu mới

Ứng dụng có thể lưu các ô đã cắt ra vào `cell_imgs.npy`. Sau đó bạn có thể dùng:

- `save_cells.py` để lưu ảnh ô số vào các thư mục nhãn
- `label_cells.py` để hỗ trợ gán nhãn thủ công
- `add_image.py` để bổ sung thêm ảnh huấn luyện

Ví dụ kiểm tra riêng dữ liệu của số 9:

```bash
python visualize_label9.py
```

## Cấu trúc thư mục

- `app.py`: giao diện Streamlit để quét và giải Sudoku
- `train_cnn.py`: huấn luyện mô hình CNN
- `engines/sudoku.py`: thuật toán giải Sudoku
- `utils/vision.py`: các hàm tiền xử lý ảnh
- `utils/helpers.py`: các hàm hỗ trợ
- `data/`: dữ liệu ảnh đã gán nhãn
- `models/`: model đã huấn luyện
- `output_cells/`: ảnh ô số xuất ra để gán nhãn

## Lưu ý

- Nên giữ ảnh gốc rõ nét, vuông góc và đủ sáng để tăng độ chính xác
- `data/`, `models/`, `output_cells/` và `cell_imgs.npy` là dữ liệu sinh ra trong quá trình làm việc, thường không cần đẩy lên GitHub
- Nếu nhận diện chưa tốt, hãy bổ sung dữ liệu thật và train lại model

## Yêu cầu môi trường

- Python 3.9+ được khuyến nghị
- Windows PowerShell hoặc terminal tương đương
- Camera để chụp bảng Sudoku khi chạy `app.py`