import os
import sys
import subprocess
from decision_tree_train import *
from random_forest_train import *
# Lista các thư viện cần thiết dựa trên các file code của bạn
REQUIRED_PACKAGES = [
    'pandas',
    'numpy',
    'scikit-learn',
    'imbalanced-learn',
    'joblib'
]


def install_and_import():
    """Tự động kiểm tra và cài đặt các thư viện chưa có."""
    print("=" * 60)
    print(" ĐANG KIỂM TRA VÀ CÀI ĐẶT CÁC THƯ VIỆN CẦN THIẾT...")
    print("=" * 60)

    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"[*] Thư viện '{package}' chưa được cài đặt. Đang tiến hành cài đặt...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"[+] Đã cài đặt thành công '{package}'.\n")
            except Exception as e:
                print(f"[!] Lỗi khi cài đặt {package}: {e}")
                sys.exit(1)

    print("[✔] Tất cả các thư viện cần thiết đã sẵn sàng!\n")


def display_header():
    """Hiển thị tiêu đề chương trình."""
    print("=" * 60)
    print("               BỘ DỰ ĐOÁN ĐỘT QUỴ            ")
    print("=" * 60)


def choose_algorithm():
    """Hỏi người dùng chọn thuật toán muốn sử dụng."""
    print("Vui lòng lựa chọn thuật toán bạn muốn dùng để dự đoán:")
    print("  1. Decision Tree (Recommended)")
    print("  2. Random Forest")
    print("  0. Thoát chương trình")
    print("-" * 60)

    choice = input("Nhập lựa chọn của bạn (1/2/0, mặc định chọn 1): ").strip()
    return choice


def run_predictor(script_name):
    """Chạy file predictor tương ứng."""
    if not os.path.exists(script_name):
        print(f"\n[!] Lỗi: Không tìm thấy file '{script_name}' trong thư mục hiện tại.")
        return

    print(f"\n---> Đang khởi chạy '{script_name}'...\n")
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except KeyboardInterrupt:
        print("\n[!] Đã dừng chương trình.")
    except Exception as e:
        print(f"\n[!] Có lỗi xảy ra khi thực thi {script_name}: {e}")


def main():
    # 1. Tải và cài đặt các thư viện thiếu
    install_and_import()

    ######################TEST_PHASE############################

    #có thể code thêm cái hỏi người dùng để người dùng tùy ý declare các thông số cho model(triển khai sau)
    model = model_decisionTree_declare(max_depth=5)
    trained_decisionTree_model = train_decisionTree_model(model)

    model2 = model_randomForest_declare()
    trained_randomForest_model = train_randomForest_model(model2)
    ############################################################

    # 2. Hiển thị tiêu đề
    display_header()

    # 3. Lựa chọn thuật toán
    choice = choose_algorithm()

    if choice == '2':
        run_predictor('random_forest_predictor.py')
    elif choice == '0':
        print("Cảm ơn bạn đã sử dụng chương trình!")
    else:
        # Mặc định hoặc chọn 1 -> Decision Tree (recommended)
        run_predictor('decision_tree_predictor.py')


if __name__ == '__main__':
    main()