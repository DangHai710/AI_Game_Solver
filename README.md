# AI Game Solver - Sudoku

Ung dung Streamlit dung OpenCV va CNN de quet anh Sudoku, cat 81 o, nhan dien chu so va giai bang Backtracking + MRV.

## Cau truc

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
├─ reports/
│  ├─ bao_cao_sudoku.md
│  └─ bao_cao_sudoku.docx
└─ requirements.txt
```

## Cai dat

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chay web app

Chay tren may tinh:

```bash
streamlit run app.py
```

Cho dien thoai cung Wi-Fi/LAN truy cap:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Mo dia chi LAN hien trong sidebar, vi du:

```text
http://192.168.x.x:8501
```

## Luong su dung

1. Tab `Chon anh de upload`: chon anh tu may tinh hoac dien thoai. Tren dien thoai, trinh duyet se cho chon chup anh moi hoac lay tu thu vien.
2. Tab `Camera truc tiep`: dung camera truc tiep neu trinh duyet cho phep. Tren dien thoai co the bi chan neu app chay HTTP LAN.
3. Sau khi co anh, app hien `Anh goc` va `81 o da cat`.
4. App tu dong luu 81 o vao `cell_imgs.npy` va `output_cells/latest/cell_r_c.png`.
5. Bam `Giai Sudoku` de hien board so da nhan dien va anh goc da dien cac so con thieu.
6. Mo `Do tin cay tung o` de xem confidence phuc vu bao cao va debug.

## Xu ly anh

- `utils/vision.py` tim contour Sudoku bang threshold thich nghi va contour lon nhat co 4 dinh.
- Bang duoc nan ve anh vuong `450x450`.
- Moi o duoc cat theo luoi 9x9, sau do lay 80% vung trung tam de tranh dinh duong vien.
- Anh o duoc threshold, loc connected component de loai nhieu, resize ve `28x28` cho CNN.

## Gan nhan data

Sau khi app tao `cell_imgs.npy`, chay:

```bash
python save_cells.py
```

Bam phim `0-9` de luu o hien tai vao `data/label_x`, bam `Space` de bo qua, `ESC` de dung.

## Train CNN

```bash
python models/train_cnn.py
```

Model moi duoc luu vao:

```text
models/cnn_sudoku.h5
```

## File chinh

- `sudoku.py`: Backtracking + MRV.
- `app.py`: giao dien Streamlit, upload/camera, nhan dien va hien ket qua.
- `utils/vision.py`: tim khung, nan anh, cat 81 o, tien xu ly o, ve loi giai len anh goc.
- `save_cells.py`: gan nhan 81 o da cat.
- `models/train_cnn.py`: train CNN tu `data/label_0..9`.
