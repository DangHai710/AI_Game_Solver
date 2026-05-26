import os
import shutil

def main():
    print("=== Thêm ảnh vào dataset số ===")
    img_path = input("Nhập đường dẫn ảnh (tương đối hoặc tuyệt đối): ").strip()
    if not os.path.isfile(img_path):
        print("Ảnh không tồn tại. Hãy kiểm tra lại đường dẫn.")
        return
    label = input("Nhập nhãn cho ảnh (0-9): ").strip()
    if not label.isdigit() or not (0 <= int(label) <= 9):
        print("Nhãn không hợp lệ. Chỉ nhận số từ 0 đến 9.")
        return
    dest_dir = os.path.join("data", f"label_{label}")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # Đặt tên file mới để tránh trùng
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    i = 1
    new_name = base_name
    while os.path.exists(os.path.join(dest_dir, new_name)):
        new_name = f"{name}_{i}{ext}"
        i += 1
    dest_path = os.path.join(dest_dir, new_name)
    shutil.copy(img_path, dest_path)
    print(f"Đã thêm ảnh vào {dest_path}")

if __name__ == "__main__":
    main()
