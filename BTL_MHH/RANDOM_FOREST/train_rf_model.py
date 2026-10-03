import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, recall_score, f1_score, roc_auc_score

# 1. Đọc dữ liệu
print(">>> Đang đọc tập dữ liệu...")
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# 2. Tiền xử lý dữ liệu sạch
# - Loại bỏ mẫu nhiễu gender = 'Other' (chỉ có 1 mẫu)
df = df[df['gender'] != 'Other']

# - Loại bỏ cột định danh 'id'
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Tách biến đặc trưng (X) và biến mục tiêu (y)
X = df.drop(columns=['stroke'])
y = df['stroke']

# Phân loại nhóm thuộc tính
numeric_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                        'work_type', 'Residence_type', 'smoking_status']

# Build Pipeline tiền xử lý
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, categorical_features)
])

# Chia tập dữ liệu Train / Test (80% / 20%) phân tầng
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Tạo Pipeline kết hợp Preprocessor + Random Forest Classifier
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,                # Số lượng cây trong rừng (200 cây)
        max_depth=10,                    # Giới hạn độ sâu để tránh overfitting
        class_weight='balanced_subsample', # Cân bằng trọng số lớp theo từng cây con
        random_state=42,
        n_jobs=-1                        # Tận dụng tối đa CPU để train nhanh hơn
    ))
])

# 4. Huấn luyện mô hình
print(">>> Đang huấn luyện Random Forest Model (200 cây)...")
rf_pipeline.fit(X_train, y_train)

# 5. Đánh giá hiệu năng trên tập Test
y_pred = rf_pipeline.predict(X_test)
y_prob = rf_pipeline.predict_proba(X_test)[:, 1]

print("\n=== KẾT QUẢ ĐÁNH GIÁ RANDOM FOREST MODEL ===")
print(f"Recall (Phát hiện ca đột quỵ): {recall_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.2f}\n")
print(classification_report(y_test, y_pred, target_names=['Không đột quỵ', 'Đột quỵ']))

# 6. Xuất Model ra file .pkl
model_filename = 'stroke_rf_model.pkl'
joblib.dump(rf_pipeline, model_filename)
print(f"\n>>> Hoàn tất! File mô hình đã được lưu tại: {model_filename}")