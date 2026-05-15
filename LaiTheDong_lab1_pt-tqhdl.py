import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Tải bộ dữ liệu Titanic có sẵn trong Seaborn
df = sns.load_dataset('titanic')
# Hiển thị 10 dòng đầu tiên
print("10 dòng dữ liệu đầu tiên:")
print(df.head(10))

# Kiểm tra số lượng bản ghi và thuộc tính
print(f"\nKích thước bộ dữ liệu: {df.shape}") # (số dòng, số cột)

# Kiểm tra kiểu dữ liệu
print("\nKiểu dữ liệu của từng cột:")
print(df.dtypes)
# Hàm describe() sẽ tính nhanh mean, min, max, std cho các cột số
print("\nThống kê cơ bản cho các cột số:")
print(df.describe())
# Thiết lập giao diện biểu đồ
sns.set_theme(style="whitegrid")

# Histogram cho thuộc tính số
plt.figure(figsize=(10, 5))
sns.histplot(df['age'].dropna(), kde=True, color='blue')
plt.title('Phân phối của Tuổi (Age)')
plt.show()

# Bar chart cho thuộc tính phân loại
plt.figure(figsize=(10, 5))
sns.countplot(x='class', data=df, palette='viridis')
plt.title('Số lượng hành khách theo Hạng vé (Class)')
plt.show()
# Kiểm tra tổng số giá trị thiếu mỗi cột
print("\nSố lượng giá trị thiếu:")
print(df.isnull().sum())

# Xử lý: Điền giá trị trung vị (median) cho cột 'age'
df['age'] = df['age'].fillna(df['age'].median())

# Xử lý: Xóa các dòng thiếu dữ liệu ở cột 'embarked' (nếu quá ít)
df.dropna(subset=['embarked'], inplace=True)
# Kiểm tra và loại bỏ bản ghi trùng lặp
print(f"\nSố dòng trùng lặp: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# Chuẩn hóa dữ liệu số (Standardization) cho cột 'fare' (giá vé)
scaler = StandardScaler()
# reshape(-1, 1) vì scaler yêu cầu mảng 2 chiều
df['fare_scaled'] = scaler.fit_transform(df[['fare']])
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Trước khi chuẩn hóa (Dữ liệu gốc)
sns.boxplot(ax=axes[0], y=df['fare'], color='salmon')
axes[0].set_title('Boxplot Giá vé (Gốc)')

# Sau khi chuẩn hóa
sns.boxplot(ax=axes[1], y=df['fare_scaled'], color='skyblue')
axes[1].set_title('Boxplot Giá vé (Đã chuẩn hóa)')

plt.tight_layout()
plt.show()
