import joblib  # <--- Thêm joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# 1: Đọc dữ liệu
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# 2: Làm sạch ban đầu, xóa ID, bỏ gender bất thường
df = df.drop(columns=['id'])
df = df[df['gender'] != 'Other']

# 3: Chia X, y và Train/Test theo 80 và 20
X = df.drop(columns=['stroke'])
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4: Phân loại cột (Đã thêm binary_cols cho hypertension và heart_disease)
num_cols_skewed = ['avg_glucose_level', 'bmi']  # bị lệch, cần trung vị, log transform, chuẩn hóa
num_cols_normal = ['age']  # ít bị lệch
binary_cols = ['hypertension', 'heart_disease']  # các biến nhị phân 0/1 giữ nguyên
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']  # chuyển sang one-hot

# 5. Pipeline tiền xử lý
num_skewed_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log_transform', FunctionTransformer(np.log1p, validate=False)),
    ('scaler', StandardScaler())
])

num_normal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num_skewed', num_skewed_pipeline, num_cols_skewed),
    ('num_normal', num_normal_pipeline, num_cols_normal),
    ('binary', 'passthrough', binary_cols),  # Giữ nguyên giá trị 0/1 của 2 cột này
    ('cat', cat_pipeline, cat_cols)
])

# 6. Fit & Transform dữ liệu
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Lấy danh sách tên cột hoàn chỉnh
encoded_cat_cols = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols))
feature_names = num_cols_skewed + num_cols_normal + binary_cols + encoded_cat_cols

# 7. Xử lý mất cân bằng bằng SMOTENC
# Cập nhật danh sách cột định danh/phân loại cho SMOTENC (bao gồm cả các cột binary và one-hot)
categorical_feature_names = binary_cols + encoded_cat_cols
cat_indices = [feature_names.index(col) for col in categorical_feature_names]

smotenc = SMOTENC(categorical_features=cat_indices, random_state=42)
X_train_resampled, y_train_resampled = smotenc.fit_resample(X_train_preprocessed, y_train)

# 8. Xuất file kết quả
train_resampled_df = pd.DataFrame(X_train_resampled, columns=feature_names)
train_resampled_df['stroke'] = y_train_resampled

test_df = pd.DataFrame(X_test_preprocessed, columns=feature_names)
test_df['stroke'] = y_test.values

train_resampled_df.to_csv('stroke_train_preprocessed.csv', index=False)
test_df.to_csv('stroke_test_preprocessed.csv', index=False)

# Lưu pkl nhằm mục đích chuẩn hóa dữ liệu cho người dùng nhập vào
joblib.dump({
    'preprocessor': preprocessor,
    'feature_names': feature_names
}, 'preprocessor.pkl')

print("Tiền xử lý, cân bằng dữ liệu và lưu preprocessor.pkl hoàn tất!")