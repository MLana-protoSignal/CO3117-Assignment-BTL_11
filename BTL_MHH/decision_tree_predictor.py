import os
import joblib
import pandas as pd


def load_resources():
    model_file = 'decision_tree_stroke_model.pkl'
    prep_file = 'preprocessor.pkl'

    if not os.path.exists(model_file):
        print(
            f"Lỗi: Không tìm thấy file mô hình '{model_file}'. Hãy chạy decision_tree_train.py!"
        )
        exit(1)
    if not os.path.exists(prep_file):
        print(
            f"Lỗi: Không tìm thấy file tiền xử lý '{prep_file}'. Hãy chạy Tien_Xu_Ly.py!"
        )
        exit(1)

    model = joblib.load(model_file)
    prep_data = joblib.load(prep_file)

    return model, prep_data['preprocessor'], prep_data['feature_names']


def get_natural_user_input():
    print("\n" + "=" * 50)
    print(" NHẬP THÔNG TIN BỆNH NHÂN (GIÁ TRỊ THỰC TẾ/TỰ NHIÊN) ")
    print("=" * 50)

    # Nhập giá trị số thực tế
    age = float(input("1. Tuổi của bệnh nhân (ví dụ: 67): "))
    avg_glucose_level = float(
        input("2. Chỉ số đường huyết trung bình (ví dụ: 228.69): ")
    )
    bmi = float(input("3. Chỉ số BMI (ví dụ: 36.6): "))

    # Nhập các tiền sử bệnh nhị phân (0 hoặc 1)
    print("\n--- Tiền sử bệnh lý ---")
    hyp_choice = input("Bệnh nhân có bị cao huyết áp không? (1. Có / 2. Không): ").strip()
    hypertension = 1 if hyp_choice == '1' else 0

    heart_choice = input("Bệnh nhân có bị bệnh tim không? (1. Có / 2. Không): ").strip()
    heart_disease = 1 if heart_choice == '1' else 0

    # Nhập giá trị danh mục tự nhiên
    print("\n--- Thông tin danh mục ---")

    g_choice = input("Giới tính (1. Female / 2. Male): ").strip()
    gender = 'Male' if g_choice == '2' else 'Female'

    m_choice = input("Đã từng kết hôn chưa? (1. Yes / 2. No): ").strip()
    ever_married = 'Yes' if m_choice == '1' else 'No'

    print("\nLoại công việc (work_type):")
    print(
        " 1. Private (Tư nhân)\n 2. Self-employed (Tự làm chủ)\n 3. Govt_job (Nhà nước)\n 4. children (Trẻ em)\n 5. Never_worked (Chưa từng làm việc)"
    )
    w_dict = {
        '1': 'Private',
        '2': 'Self-employed',
        '3': 'Govt_job',
        '4': 'children',
        '5': 'Never_worked',
    }
    work_type = w_dict.get(input("Chọn (1-5): ").strip(), 'Private')

    r_choice = input("\nNơi sinh sống (1. Urban / 2. Rural): ").strip()
    Residence_type = 'Urban' if r_choice == '1' else 'Rural'

    print("\nTình trạng hút thuốc (smoking_status):")
    print(
        " 1. never smoked (Chưa bao giờ hút)\n 2. formerly smoked (Đã từng hút)\n 3. smokes (Đang hút)\n 4. Unknown (Không rõ)"
    )
    s_dict = {
        '1': 'never smoked',
        '2': 'formerly smoked',
        '3': 'smokes',
        '4': 'Unknown',
    }
    smoking_status = s_dict.get(input("Chọn (1-4): ").strip(), 'never smoked')

    # Đóng gói dữ liệu thô dạng DataFrame ban đầu (đầy đủ các cột đầu vào cho preprocessor)
    raw_df = pd.DataFrame([
        {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status,
        }
    ])

    return raw_df


def predict_stroke():
    model, preprocessor, feature_names = load_resources()

    while True:
        # 1. Lấy dữ liệu tự nhiên từ người dùng
        raw_df = get_natural_user_input()

        # 2. Dùng chính xác preprocessor đã fit ở file Tien_Xu_Ly.py để biến đổi
        processed_array = preprocessor.transform(raw_df)

        # 3. Tạo DataFrame khớp đúng danh sách cột đặc trưng
        input_df = pd.DataFrame(processed_array, columns=feature_names)

        # 4. Dự đoán
        prediction = model.predict(input_df)[0]

        print("\n" + "*" * 45)
        if prediction == 1:
            print(" KẾT QUẢ DỰ ĐOÁN: Nguy cơ CAO bị đột quỵ (Stroke = 1)")
        else:
            print(" KẾT QUẢ DỰ ĐOÁN: Nguy cơ THẤP / KHÔNG đột quỵ (Stroke = 0)")
        print("*" * 45)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            print(f"Xác suất không đột quỵ (0): {probs[0]*100:.2f}%")
            print(f"Xác suất có đột quỵ (1):    {probs[1]*100:.2f}%")

        print("\n" + "-" * 45)
        cont = (
            input("Tiếp tục dự đoán cho bệnh nhân khác? (y/n): ")
            .strip()
            .lower()
        )
        if cont != 'y':
            print("Cảm ơn bạn đã sử dụng chương trình!")
            break


if __name__ == '__main__':
    predict_stroke()