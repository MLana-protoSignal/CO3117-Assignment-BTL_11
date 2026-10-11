import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


def model_declare(max_depth=5, min_samples_leaf=10, random_state=42):
    # Khởi tạo và trả về model trực tiếp từ các tham số truyền vào
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    return model

def train_model(model_input):
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
    
    # 4. Hỏi tên file lưu model (mặc định 'decision_tree_stroke_model.pkl')
    model_name_input = input("Nhập tên file để lưu model (mặc định 'decision_tree_stroke_model.pkl'): ").strip()
    if not model_name_input:
        save_filename = 'decision_tree_stroke_model.pkl'
    else:
        # Tự động thêm đuôi .pkl nếu người dùng không nhập
        save_filename = model_name_input if model_name_input.endswith('.pkl') else f"{model_name_input}.pkl"

    # 5. Lưu mô hình
    joblib.dump(model_input, save_filename)
    print(f"-> Đã lưu mô hình thành công vào '{save_filename}'!")
    
    return model_input

