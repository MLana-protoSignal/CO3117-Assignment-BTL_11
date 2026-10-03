import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, recall_score, f1_score

# 1. Đọc dữ liệu
print(">>> Đang đọc tập dữ liệu...")
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# 2. Tiền xử lý sạch dữ liệu
# - Loại bỏ hàng có gender = 'Other' (chỉ có 1 mẫu)
df = df[df['gender'] != 'Other']

# - Loại bỏ cột định danh 'id'
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Tách biến đặc trưng (X) và biến mục tiêu (y)
X = df.drop(columns=['stroke'])
y = df['stroke']

# Phân loại nhóm đặc trưng
numeric_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                        'work_type', 'Residence_type', 'smoking_status']

# Build Pipeline tiền xử lý
# - Biến số: Điền N/A bằng Median + Chuẩn hóa StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# - Biến phân loại: Mã hóa One-Hot
cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Kết hợp preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, categorical_features)
])

# Chia tập dữ liệu Train / Test (80% / 20%) có bảo toàn tỷ lệ lớp (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Tạo Pipeline hoàn chỉnh kết hợp Preprocessor + Decision Tree
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        random_state=42,
        max_depth=5,                 # Giới hạn độ sâu để tránh overfitting
        class_weight='balanced'      # Xử lý mất cân bằng lớp (Severe Class Imbalance)
    ))
])

# 4. Huấn luyện mô hình
print(">>> Đang huấn luyện Decision Tree Model...")
full_pipeline.fit(X_train, y_train)

# 5. Đánh giá hiệu năng trên tập Test
y_pred = full_pipeline.predict(X_test)
print("\n=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===")
print(f"Recall (Khả năng phát hiện ca đột quỵ): {recall_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}\n")
print(classification_report(y_test, y_pred, target_names=['Không đột quỵ', 'Đột quỵ']))

# 6. Xuất Model ra file .pkl
model_filename = 'stroke_dt_model.pkl'
joblib.dump(full_pipeline, model_filename)
print(f"\n>>> Hoàn tất! File mô hình đã được xuất ra: {model_filename}")