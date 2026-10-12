# Cách dùng chương trình đã huấn luyện sẵn để dự đoán: 
Chạy main.py bằng cú pháp python main.py

# Nếu muốn huấn luyện dữ liệu mới:
 Chạy Tien_Xu_Ly.py, sau đó chọn chạy decision_tree_train.py hoặc random_forest_train.py. Lúc này chạy main.py sẽ dùng model được huấn luyện mới.

# Patch Notes:
 Do hiện tại mới học tới decision tree, cho nên mô hình này chạy tốt. Còn random forest thì dự đoán khá là kém chính xác, đây cũng là một điểm cần được lý giải thêm nếu cho vào báo cáo.

# BONUS:
Trong thư mục BieuDoPhanTich, tôi có viết một file dùng để vẽ các biểu đồ thông số giúp cho quá trình phân tích dữ liệu trực quan hơn, để các thành viên làm phần nội dung có thể lý giải các hiện tượng với mô hình và cho việc đánh báo cáo. 

# PATCH 1.1.1: cấu trúc lại hệ thống chương trình
## Tái cấu trúc lại file train, trong đó hàm model_decisionTree_declare() nhằm mục đích khai báo model 
Ví dụ cách dùng: my_model = model_decisionTree_declare(max_depth=5, min_samples_leaf=10, random_state=42)
Hướng tới file main: trong main sẽ viết hỏi các thông số maxdepth,... từ bàn phím người dùng và truyền cho model_decisionTree_declare

Sau đó để huấn luyện, ví dụ: trained_model = train_decisionTree_model(my_model)

## Về vấn đề predict:
### Giả sử muốn redict một tập test luôn cho lẹ thì ví dụ cách dùng:

model = model_decisionTree_declare(max_depth=5)

trained_decisionTree_model = train_decisionTree_model(model)

df_test = pd.read_csv("stroke_test.csv")

X_test = df_test.drop(columns=["stroke"])

y_test = df_test["stroke"]

y_pred = trained_decisionTree_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

### Muốn test lẻ để tùy ý nhập thông số thì gọi file predict(chỉ là tính năng để demo lẻ thôi)

## Đã thay đổi file tiền xử lý lại, chỉ có tiền xử lý file và quăng ra stroke_preprocessed.csv

## hiện chưa hoàn chỉnh lại predictor cho cả 2 mô hình, sẽ hoàn thành sau và cập nhật

