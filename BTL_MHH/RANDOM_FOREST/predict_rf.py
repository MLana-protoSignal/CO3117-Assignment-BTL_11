import pandas as pd
import joblib
import sys

# Load model đã huấn luyện
try:
    model = joblib.load('stroke_rf_model.pkl')
except FileNotFoundError:
    print("LỖI: Không tìm thấy file 'stroke_rf_model.pkl'. Hãy chạy file 'train_rf_model.py' trước!")
    sys.exit()

def get_choice(prompt, valid_options):
    """Hàm hỗ trợ nhập lựa chọn hợp lệ"""
    while True:
        print(prompt)
        for k, v in valid_options.items():
            print(f"  {k}. {v}")
        choice = input("Nhập lựa chọn của bạn: ").strip()
        if choice in valid_options:
            return valid_options[choice]
        print("-> Lựa chọn không hợp lệ, vui lòng nhập lại!\n")

def get_float(prompt, min_val=0.0, max_val=300.0):
    """Hàm hỗ trợ nhập số thực hợp lệ"""
    while True:
        try:
            val = float(input(prompt).strip())
            if min_val <= val <= max_val:
                return val
            print(f"-> Giá trị phải nằm trong khoảng [{min_val} - {max_val}]!")
        except ValueError:
            print("-> Vui lòng nhập một số hợp lệ!")

print("="*60)
print("   HỆ THỐNG DỰ ĐOÁN NGUY CƠ ĐỘT QUẠ (RANDOM FOREST MODEL)")
print("="*60)
print("Vui lòng nhập thông tin bệnh nhân:\n")

# 1. Nhập các thuộc tính
gender = get_choice("1. Giới tính:", {'1': 'Male', '2': 'Female'})
age = get_float("2. Tuổi bệnh nhân (0 - 120): ", min_val=0, max_val=120)

hypertension_str = get_choice("3. Tiền sử cao huyết áp:", {'0': 'Không', '1': 'Có'})
hypertension = 1 if hypertension_str == 'Có' else 0

heart_disease_str = get_choice("4. Tiền sử bệnh tim:", {'0': 'Không', '1': 'Có'})
heart_disease = 1 if heart_disease_str == 'Có' else 0

ever_married = get_choice("5. Tình trạng kết hôn:", {'1': 'Yes', '2': 'No'})

work_type = get_choice("6. Loại hình công việc:", {
    '1': 'Private',
    '2': 'Self-employed',
    '3': 'Govt_job',
    '4': 'children',
    '5': 'Never_worked'
})

Residence_type = get_choice("7. Khu vực sinh sống:", {'1': 'Urban', '2': 'Rural'})
avg_glucose_level = get_float("8. Chỉ số đường huyết trung bình (avg_glucose_level): ", min_val=30, max_val=350)
bmi = get_float("9. Chỉ số khối cơ thể (BMI): ", min_val=10, max_val=100)

smoking_status = get_choice("10. Tình trạng hút thuốc:", {
    '1': 'never smoked',
    '2': 'formerly smoked',
    '3': 'smokes',
    '4': 'Unknown'
})

# 2. Đóng gói dữ liệu đầu vào
patient_data = pd.DataFrame([{
    'gender': gender,
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'ever_married': ever_married,
    'work_type': work_type,
    'Residence_type': Residence_type,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': smoking_status
}])

# 3. Dự đoán kết quả & Xác suất
probabilities = model.predict_proba(patient_data)[0]
stroke_risk_prob = probabilities[1] * 100

# Áp dụng ngưỡng cảnh báo y tế (Clinical Threshold: 15%)
CLINICAL_THRESHOLD = 0.15
prediction = 1 if probabilities[1] >= CLINICAL_THRESHOLD else 0

print("\n" + "="*60)
print("               KẾT QUẢ DỰ ĐOÁN (RANDOM FOREST)")
print("="*60)
print(f"-> Xác suất nguy cơ đột quỵ ước tính: {stroke_risk_prob:.2f}%")
print(f"-> Xác suất an toàn: {probabilities[0]*100:.2f}%\n")

if prediction == 1:
    print("-> ĐÁNH GIÁ: CẢNH BÁO NGUY CƠ ĐỘT QUỴ (Lớp 1)")
    print("   (Bệnh nhân có nhiều yếu tố nguy cơ tích lũy vượt ngưỡng an toàn y tế 15%)")
else:
    print("-> ĐÁNH GIÁ: NGUY CƠ THẤP (Lớp 0)")
    print("   (Bệnh nhân nằm trong vùng an toàn)")
print("="*60)