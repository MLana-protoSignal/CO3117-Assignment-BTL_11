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


def train_random_forest():
    train_file = 'stroke_train_preprocessed.csv'
    test_file = 'stroke_test_preprocessed.csv'

    # 1. Kiểm tra sự tồn tại của file dữ liệu đã tiền xử lý
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(
            f"Lỗi: Không tìm thấy '{train_file}' hoặc '{test_file}'. Hãy chạy Tien_Xu_Ly.py trước!"
        )
        return

    print("--> Đang tải dữ liệu huấn luyện và kiểm thử...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Chia X, y cho tập train và test
    X_train = train_df.drop(columns=['stroke'])
    y_train = train_df['stroke']

    X_test = test_df.drop(columns=['stroke'])
    y_test = test_df['stroke']

    # 2. Khởi tạo và huấn luyện mô hình Random Forest
    print("--> Đang huấn luyện mô hình Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,          
        min_samples_split=2,
        criterion='entropy',    
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 3. Đánh giá mô hình trên tập test
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST ")
    print("=" * 50)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print("\nMa trận nhầm lẫn (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))
    print("\nBáo cáo chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred))

    # 4. Lưu mô hình đã huấn luyện
    model_filename = 'random_forest_stroke_model.pkl'
    joblib.dump(model, model_filename)
    print("=" * 50)
    print(f"--> Đã lưu mô hình thành công vào file: '{model_filename}'")


if __name__ == '__main__':
    train_random_forest()