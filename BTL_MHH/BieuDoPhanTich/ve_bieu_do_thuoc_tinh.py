import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cấu hình giao diện chung
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
palette = 'Set2'

# 2. Đọc dữ liệu
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# --- 1. BIỂU ĐỒ BIẾN MỤC TIÊU (stroke) ---
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x='stroke', palette=palette)
plt.title('1. Phân Bố Biến Mục Tiêu (stroke)', fontsize=12, fontweight='bold')
plt.xlabel('Stroke (0: Không, 1: Có)')
plt.ylabel('Số lượng')
plt.tight_layout()
plt.savefig('stroke_01_stroke.png', dpi=300)
plt.close()

# --- 2. CÁC BIẾN PHÂN LOẠI (Categorical Features) ---
cat_cols = [
    ('gender', '2. Giới Tính (gender) theo Stroke'),
    ('hypertension', '3. Cao Huyết Áp (hypertension) theo Stroke'),
    ('heart_disease', '4. Bệnh Tim (heart_disease) theo Stroke'),
    ('ever_married', '5. Tình Trạng Hôn Nhân (ever_married) theo Stroke'),
    ('work_type', '6. Loại Hình Công Việc (work_type) theo Stroke'),
    ('Residence_type', '7. Khu Vực Sinh Sống (Residence_type) theo Stroke'),
    ('smoking_status', '8. Tình Trạng Hút Thuốc (smoking_status) theo Stroke')
]

for col, title in cat_cols:
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=col, hue='stroke', palette=palette)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel(col)
    plt.ylabel('Số lượng')
    if col in ['work_type', 'smoking_status']:
        plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f'stroke_{col}.png', dpi=300)
    plt.close()

# --- 3. CÁC BIẾN SỐ / LIÊN TỤC (Numerical Features) ---
# Tuổi (age) - Boxplot
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x='stroke', y='age', palette=palette)
plt.title('9. Phân Bố Tuổi (age) theo Stroke', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('stroke_age.png', dpi=300)
plt.close()

# Mức đường huyết (avg_glucose_level) - KDE Plot
plt.figure(figsize=(7, 5))
sns.kdeplot(data=df[df['stroke']==0]['avg_glucose_level'], label='Không đột quỵ (0)', fill=True, color='#66c2a5')
sns.kdeplot(data=df[df['stroke']==1]['avg_glucose_level'], label='Đột quỵ (1)', fill=True, color='#fc8d62')
plt.title('10. Phân Bố Mức Đường Huyết (avg_glucose_level) theo Stroke', fontsize=12, fontweight='bold')
plt.xlabel('avg_glucose_level')
plt.legend()
plt.tight_layout()
plt.savefig('stroke_avg_glucose_level.png', dpi=300)
plt.close()

# Chỉ số BMI (bmi) - Boxplot
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x='stroke', y='bmi', palette=palette)
plt.title('11. Phân Bố Chỉ Số BMI (bmi) theo Stroke', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('stroke_bmi.png', dpi=300)
plt.close()

# --- 4. MA TRẬN TƯƠNG QUAN (Correlation Heatmap) ---
plt.figure(figsize=(7, 6))
num_cols = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']]
sns.heatmap(num_cols.corr(), annot=True, fmt='.2f', cmap='Blues')
plt.title('12. Ma Trận Tương Quan Chi Tiết', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('stroke_12_correlation_matrix.png', dpi=300)
plt.close()

print("Hoàn tất xuất tất cả các biểu đồ thành các file ảnh PNG riêng biệt!")