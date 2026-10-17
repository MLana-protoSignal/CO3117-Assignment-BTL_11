import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Đọc dữ liệu
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Làm sạch ban đầu, xóa ID, bỏ gender bất thường
df = df.drop(columns=['id'])
df = df[df['gender'] != 'Other']

# Tách X và y 
X = df.drop(columns=['stroke'])
y = df['stroke']

# Phân loại cột
num_cols_skewed = ['avg_glucose_level', 'bmi']  # bị lệch, cần trung vị, log transform, chuẩn hóa
num_cols_normal = ['age']  # ít bị lệch
binary_cols = ['hypertension', 'heart_disease']  # các biến nhị phân 0/1 giữ nguyên
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']  # chuyển sang one-hot

# Pipeline tiền xử lý
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

# Fit và transform toàn bộ dữ liệu x
X_preprocessed = preprocessor.fit_transform(X)

# lấy danh sách tên cột hoàn chỉnh
encoded_cat_cols = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols))
feature_names = num_cols_skewed + num_cols_normal + binary_cols + encoded_cat_cols

# xử lý mất cân bằng bằng SMOTE trên toàn bộ dữ liệu
categorical_feature_names = binary_cols + encoded_cat_cols
cat_indices = [feature_names.index(col) for col in categorical_feature_names]

smotenc = SMOTENC(categorical_features=cat_indices, random_state=42)
X_resampled, y_resampled = smotenc.fit_resample(X_preprocessed, y)

# xuất file kết quả duy nhất
resampled_df = pd.DataFrame(X_resampled, columns=feature_names)
resampled_df['stroke'] = y_resampled.values if hasattr(y_resampled, 'values') else y_resampled

resampled_df.to_csv('stroke_preprocessed.csv', index=False)

# lưu pkl nhằm mục đích chuẩn hóa dữ liệu cho người dùng nhập vào
# khu vực này chỉ nhằm mục đích cho triển trai predictor về sau 
joblib.dump({
    'preprocessor': preprocessor,
    'feature_names': feature_names
}, 'preprocessor.pkl')

