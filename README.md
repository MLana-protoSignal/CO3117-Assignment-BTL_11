# Machine Learning Assignment (English below)
## Thông tin nhóm

Học kì: 261  
Môn học: Học máy (CO3117)  
Lớp: L03  
Nhóm: BTL_11  
Giảng viên hướng dẫn: Võ Thanh Hùng

---
## Sinh viên thực hiện

1. Nguyễn Mạnh Lân - 2411855
2. Trần Ngọc Luân - 2411980
3. Nguyễn Tuấn Mạnh - 2412017
4. Trần Thanh Hưng Thịnh - 2413345

---
## Mô tả chạy chương trình
### Chạy file main.py (Tương đương với double-click vào file run_program.bat ở hệ điều hành Windows)

Khi chạy file main.py, hệ thống sẽ hiện thực một chương trình cho phép người dùng tương tác với hệ thống để thực hiện dự đoán và đánh giá hiệu năng của các giải thuật học máy. Chương trình được chạy sẽ đảm bảo đủ các bước:
1. Kiểm tra và cài đặt thư viện: Ở bước này, chương trình sẽ tự động kiểm tra các thư viện cần thiết để chạy chương trình. Nếu chương trình phát hiện người dùng chưa cài một hoặc nhiều thư viện cần thiết thì tự động tiến hành cài thư viện đó.
2. Tiền xử lý: Ở bước này, chương trình yêu cầu người dùng nhập đường dẫn của file dataset gốc (file có đuôi .csv), người dùng có thể nhấn Enter để bỏ qua bước này và chương trình sẽ lấy file healthcare-dataset-stroke-data.csv có sẵn. Sau khi thực hiện tiền xử lý xong, chương trình tự động lưu data đã tiền xử lý vào file stroke_preprocessed.csv.
3. Khai báo và huấn luyện mô hình với các tham số mặc định tối ưu: Ở bước này, chương trình yêu cầu người dùng nhập đường dẫn của file data để huấn luyện mô hình và đường dẫn của file Python Pickle (file có đuôi .pkl) để lưu mô hình sau khi đã huấn luyện. Việc nhập path dẫn tới 2 file này chỉ thực hiện khi có các ý tưởng mở rộng mô hình sau này, nên để sử dụng hệ thống mặc định, người dùng chỉ cần nhấn Enter để bỏ qua bước nhập đường dẫn này.
4. Tương tác với người dùng: Ở bước này, chương trình rơi vào vòng lặp vô hạn, cho phép người dùng triển khai các yêu cầu tương tác với hệ thống:
    * Process = 0: Chương trình in ra thông báo kết thúc, đồng thời kết thúc vòng lặp và chương trình kết thúc.
    * Process = 1: Chương trình yêu cầu người dùng chọn giải thuật học máy (Decision Tree hoặc Random Forest) và chạy file python predictor tương ứng. Khi này, chương trình cần 2 file decision_tree_stroke_model.pkl và random_forest_stroke_model.pkl để có thể dự đoán dựa trên mô hình với các tham số mặc định tối ưu đã được huấn luyện ở bước 3. 
    * Process = 2: Chương trình yêu cầu người dùng chọn giải thuật học máy (Decision Tree hoặc Random Forest hoặc chạy cả 2) và sau đó sẽ chọn loại model để tiến hành đánh giá hiệu năng trên file data gốc healthcare-dataset-stroke-data.csv có sẵn. Với Default model, chương trình sẽ tự động lấy chính model đã khai báo ở bước 3 để tiến hành đánh giá. Với Input model, người dùng phải tự nhập các tham số để tiến hành đánh giá và người dùng có thể nhấn Enter để sử dụng tham số mặc định (default). Ở các bước yêu cầu lựa chọn như chọn giải thuật học máy hay chọn loại model, nếu người dùng chọn 0 thì chương trình sẽ đưa người dùng về phần lựa chọn Process.
---
### Chạy file performance_evaluation.py (Tương đương với double-click vào file run_performance_evaluation.bat ở hệ điều hành Windows)

Khi chạy file performance_evaluation.py, chương trình được chạy sẽ chạy lại các kịch bản khác nhau để giúp người dùng so sánh các trường hợp thay đổi tham số cũng như so sánh hiệu quả của 2 mô hình khi tham số thay đổi.

---
### Chạy file hyperparameter_tuning.py (Tương đương với double-click vào file run_hyperparameter_tuning.bat ở hệ điều hành Windows)

Khi chạy file hyperparameter_tuning.py, chương trình được chạy sẽ tiến hành kỹ thuật tối ưu tham số (hyperparameter tuning). Chương trình lúc này sẽ rơi vào hàng loạt kịch bản để tìm ra tham số tối ưu, do đó, khi chạy chương trình này sẽ mất khoảng 10-15 phút để có thể chạy và in ra tất cả các thông số và đánh giá cần thiết.

---
## Group Information

Semester: 261  
Course: Machine Learning (CO3117)  
Class: L03  
Group: BTL_11  
Instructor: Võ Thanh Hùng

---
## Student

1. Nguyễn Mạnh Lân - 2411855
2. Trần Ngọc Luân - 2411980
3. Nguyễn Tuấn Mạnh - 2412017
4. Trần Thanh Hưng Thịnh - 2413345

---
## Program execution description
### Run the main.py file (Equivalent to double-clicking the run_program.bat file on the Windows operating system)

When the main.py file is executed, the system launches a program that allows user to interact with it in order to perform predictions and evaluate the performance of machine learning algorithms. The program will excute with below stages:
1. Checking and installing libraries: At this stage, the program will automatically check for the libraries required to run. If it detects that one or more necessary libraries are missing, it will proceed to install them automatically.
2. Preprocessing: At this stage, the program requires the user to enter the path to the original dataset file (file with .csv extension). User can press Enter to skip this step, and program will use healthcare-dataset-stroke-data.csv file in the same folder. When the preprocessing is complete, the program automatically saves the preprocessed data into stroke_preprocessed.csv file.
3. Defining and training the model with optimal default parameters: At this stage, the program requires the user to enter the path to traning data file. Then, the program requires the user to enter the path to Python Pickle file (file with .pkl extension) to save the trained models. Entering the paths in this stage is only necessary if user plan to extend the model in the future. Therefore, to use the system with its default settings, user can simply press Enter to skip entering paths.
4. User interaction: At this stage, the program will enter an infinite loop and allow the user to execute tasks that interact with the system:
    * Process = 0: The program prints a termination message, exits the loop, and then terminates.
    * Process = 1: The program requires user to select machine learning algorithm (Decision Tree or Random Forest) and run the corresponding Python predictor file. The program needs 2 file with .pkl extension (decision_tree_stroke_model.pkl and random_forest_stroke_model.pkl) to perform predictions based on the model with the optimal default parameters that were trained in stage 3.
    * Process = 2: The program requires user to select machine learning algorithm (Decision Tree or Random Forest or both) and model to perform the evaluation on the origin dataset file (healthcare-dataset-stroke-data.csv). With Default model, the program will automatically use the very model declared in stage 3 to perform the evaluation. With Input model, user must manually enter the parameters to perform the evaluation, and user can press Enter to use default parameters. For steps that require user to select a machine learning algorithm or a model type, if user selects 0, the program will direct the user to the Process selection step.
---
### Run the performance_evaluation.py file (Equivalent to double-clicking the run_performance_evaluation.bat file on the Windows operating system)

When running the performance_evaluation.py file, the program then runs various scenarios and allowing user to compare different parameter settings and two machine learning algorithms with the same parameters.

---
### Run the hyperparameter_tuning.py file (Equivalent to double-clicking the run_hyperparameter_tuning.bat file on the Windows operating system)

When running the hyperparameter_tuning.py file, the program will perform the hyperparameter tuning technique. The program will iterate through a series of scenarios to determine the optimal parameters, and the program will take approximately 10–15 minutes to complete and output all the necessary parameters and evaluations.