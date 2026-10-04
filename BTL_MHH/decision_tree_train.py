import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


def train_and_save_model():
    # Đọc dữ liệu đã tiền xử lý từ Tien_Xu_Ly.py
    train_path = 'stroke_train_preprocessed.csv'
    test_path = 'stroke_test_preprocessed.csv'

    print("Đang tải dữ liệu huấn luyện...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    target_col = 'stroke'
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # Huấn luyện mô hình Decision Tree
    print("Đang huấn luyện Decision Tree...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Đánh giá mô hình
    y_pred = model.predict(X_test)
    print("\n=== ĐÁNH GIÁ MÔ HÌNH THÀNH CÔNG ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Lưu duy nhất mô hình
    joblib.dump(model, 'decision_tree_stroke_model.pkl')
    print("-> Đã lưu mô hình thành công vào 'decision_tree_stroke_model.pkl'!")


if __name__ == "__main__":
    train_and_save_model()