import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def model_randomForest_declare(n_estimators=200, max_depth=None, min_samples_split=2, criterion='entropy', random_state=42, n_jobs=-1):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,          
        min_samples_split=min_samples_split,
        criterion=criterion,    
        random_state=random_state,
        n_jobs=n_jobs
    )
    return model

def train_randomForest_model(model_input):
    # 1. Hỏi người dùng nhập đường dẫn file train (mặc định 'stroke_preprocessed.csv')
    path_input = input("Nhập đường dẫn file train (mặc định 'stroke_preprocessed.csv'): ").strip()
    train_path = path_input if path_input else 'stroke_preprocessed.csv'

    # 2. Đọc dữ liệu từ đường dẫn
    df_train = pd.read_csv(train_path)

    target_col = 'stroke'
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    # 3. Huấn luyện mô hình truyền vào
    model_input.fit(X_train, y_train)
    
    # 4. Hỏi tên file lưu model (mặc định 'randomForest_stroke_model.pkl')
    model_name_input = input("Nhập tên file để lưu model (mặc định 'randomForest_stroke_model.pkl'): ").strip()
    if not model_name_input:
        save_filename = 'randomForest_stroke_model.pkl'
    else:
        # Tự động thêm đuôi .pkl nếu người dùng không nhập
        save_filename = model_name_input if model_name_input.endswith('.pkl') else f"{model_name_input}.pkl"

    # 5. Lưu mô hình
    joblib.dump(model_input, save_filename)
    print(f"-> Đã lưu mô hình thành công vào '{save_filename}'!")
    
    return model_input