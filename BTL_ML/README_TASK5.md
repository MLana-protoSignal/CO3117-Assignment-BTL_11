    Cách chạy
    python Tien_Xu_Ly.py              # tạo stroke_train/test_preprocessed.csv và preprocessor.pkl
    python decision_tree_train.py     # baseline Decision Tree (task 3)
    python random_forest_train.py     # baseline Random Forest (task 3)
    python hyperparameter_tuning.py   # task 5, khoảng 3-10 phút tùy máy
    
    
    Kết quả được lưu trong thư mục KetQuaTuning/. Hai mô hình sau khi tune được lưu cạnh hai mô hình baseline: decision_tree_tuned_model.pkl và random_forest_tuned_model.pkl của Mạnh.
    Hai file này nhận đầu vào giống baseline, nên mình muốn dự đoán bằng mô hình tuned chỉ cần đổi tên file mô hình trong load_resources() của *_predictor.py.
    
    
    Baseline là hai mô hình đã lưu ở task 3 (decision_tree_stroke_model.pkl, random_forest_stroke_model.pkl).
    
    Không cross-validation trực tiếp trên stroke_train_preprocessed.csv: file này đã bị SMOTENC sẵn nên mẫu tổng hợp và mẫu thật "họ hàng" của nó rơi vào cả fold train lẫn fold validation, làm điểm CV cao giả tạo (rò rỉ dữ liệu). 
    Script dựng lại tập train gốc (chia giống hệt Tien_Xu_Ly.py) và đặt tiền xử lý -> SMOTENC -> mô hình trong một imblearn.pipeline.Pipeline, nên mỗi fold chỉ SMOTE trên phần train của fold đó. Phần đầu của output in bảng so sánh điểm F1 giữa cách làm sai, cách làm đúng và test thật.
    
    Decision Tree: GridSearchCV (2 x 8 x 5 = 80 cấu hình, 5-fold) trên criterion, max_depth, min_samples_leaf.
    
    Random Forest: RandomizedSearchCV (30 cấu hình lấy ngẫu nhiên, 5-fold) trên n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_features.
    
    Metric chọn tham số: F1 của lớp stroke (REFIT_METRIC, có thể đổi sang average_precision hoặc recall). Không dùng accuracy vì chỉ khoảng 5% mẫu là stroke. Không tune class_weight vì SMOTENC đã cân bằng 50/50 trong từng fold train.
    
    *Tổng kết: tập test (stroke_test_preprocessed.csv) chỉ dùng một lần để so sánh baseline và tuned, không dùng để chọn tham số. Mọi random_state đều là 42 nên khi t chạy lại cho giống kết quả.
