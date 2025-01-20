import os
import pandas as pd
import shutil
from tqdm import tqdm

# Đường dẫn file CSV và thư mục gốc
csv_file_path = r'E:\Codes\full-stack-on-prem-cv-mlops\datasets\Images\annotation_df.csv'  # Cập nhật đường dẫn chính xác
images_root_dir = r'E:\Codes\full-stack-on-prem-cv-mlops\datasets\Images\images'  # Thư mục chứa ảnh gốc
output_csv_file_path = r'E:\Codes\full-stack-on-prem-cv-mlops\datasets\Images\annotation_df_valid.csv'  # Đường dẫn để lưu file CSV mới

# Kiểm tra sự tồn tại của file CSV
if not os.path.exists(csv_file_path):
    print(f"Tệp CSV không tồn tại: {csv_file_path}")
    exit()

# Đọc file CSV
df = pd.read_csv(csv_file_path)

# Chuẩn hóa đường dẫn trong cột 'abs_path' để sử dụng
df['abs_path'] = df['abs_path'].str.replace('/', '\\').str.lstrip('\\')

# Tạo danh sách các hàng hợp lệ
valid_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_name = row['image_name']  # Tên ảnh trong cột image_name
    abs_path = row['abs_path']      # Đường dẫn chuẩn hóa trong cột abs_path

    # Tìm ảnh gốc trong thư mục images
    found = False
    for root, _, files in os.walk(images_root_dir):
        if image_name in files:
            src_path = os.path.join(root, image_name)  # Đường dẫn ảnh gốc
            found = True
            break

    if not found:
        print(f"Không tìm thấy ảnh: {image_name}")
        continue

    # Thêm dòng hợp lệ vào danh sách
    valid_rows.append(row)

    # Tạo thư mục đích nếu chưa tồn tại
    new_dir = os.path.dirname(abs_path)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Sao chép ảnh đến thư mục mới
    shutil.copy2(src_path, abs_path)
    print(f"Đã sao chép: {src_path} -> {abs_path}")

# Tạo DataFrame mới từ danh sách hợp lệ
valid_df = pd.DataFrame(valid_rows)

# Khôi phục đường dẫn trong cột 'abs_path' về dạng ban đầu, thêm dấu '/' ở đầu
valid_df['abs_path'] = valid_df['abs_path'].str.replace('\\', '/')
valid_df['abs_path'] = valid_df['abs_path'].apply(lambda x: f"/{x}" if not x.startswith('/') else x)

# Lưu file CSV mới chỉ chứa các file tồn tại
valid_df.to_csv(output_csv_file_path, index=False)
print(f"Đã lưu file annotation_df mới tại: {output_csv_file_path}")
