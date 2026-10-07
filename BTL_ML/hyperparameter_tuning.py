

import json
import os
import time

import joblib
import matplotlib

matplotlib.use('Agg')  # chỉ lưu ảnh ra file, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier

# ============================== CẤU HÌNH ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'KetQuaTuning')

RANDOM_STATE = 42        # phải trùng với Tien_Xu_Ly.py để chia train/test giống hệt
N_SPLITS = 5             # số fold của cross-validation
N_ITER_RF = 30           # số cấu hình Random Forest được thử (RandomizedSearchCV)
N_JOBS = 1              # dùng hết nhân CPU (đặt 1 nếu máy yếu hoặc gặp lỗi đa tiến trình)
RUN_LEAKAGE_DEMO = True  # in bảng minh họa rò rỉ dữ liệu (mục 0)

# Metric dùng để CHỌN tham số tốt nhất. Dữ liệu mất cân bằng (~5% stroke) nên không dùng accuracy.
# F1 của lớp stroke cân bằng giữa bỏ sót ca bệnh (recall) và báo nhầm (precision).
# Có thể đổi thành 'average_precision' (PR-AUC) hoặc 'recall'.
REFIT_METRIC = 'f1'
SCORING = {  # các metric được ghi lại trong mỗi lần cross-validation
    'f1': 'f1',
    'recall': 'recall',
    'precision': 'precision',
    'average_precision': 'average_precision',
    'roc_auc': 'roc_auc',
}

# Không tune class_weight vì SMOTENC đã cân bằng 50/50 trong từng fold train.
DT_GRID = {  # 2 x 8 x 5 = 80 cấu hình -> đủ nhỏ để duyệt hết bằng GridSearchCV
    'clf__criterion': ['gini', 'entropy'],
    'clf__max_depth': [3, 4, 5, 6, 8, 10, 15, None],
    'clf__min_samples_leaf': [1, 5, 10, 20, 50],
}
RF_DIST = {  # 2160 cấu hình -> quá lớn để duyệt hết, chỉ lấy ngẫu nhiên N_ITER_RF cấu hình
    'clf__n_estimators': [100, 200, 300, 500],
    'clf__criterion': ['gini', 'entropy'],
    'clf__max_depth': [5, 8, 10, 15, 20, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4, 8, 16],
    'clf__max_features': ['sqrt', 'log2', 0.5],
}

NUMERIC_FEATURES = ['avg_glucose_level', 'bmi', 'age']  # cột số trong Tien_Xu_Ly.py (phần còn lại là 0/1)
BASELINE_FILES = {
    'Decision Tree': 'decision_tree_stroke_model.pkl',
    'Random Forest': 'random_forest_stroke_model.pkl',
}
TUNED_FILES = {
    'Decision Tree': 'decision_tree_tuned_model.pkl',
    'Random Forest': 'random_forest_tuned_model.pkl',
}


def path(name):
    return os.path.join(BASE_DIR, name)


def need_file(filename, hint):
    if not os.path.exists(path(filename)):
        print(f"Lỗi: Không tìm thấy '{filename}'. {hint}")
        raise SystemExit(1)


# ============================== DỮ LIỆU ==============================
def load_data():
    need_file('healthcare-dataset-stroke-data.csv', 'Hãy đặt file dữ liệu gốc cùng thư mục với script.')
    for f in ['preprocessor.pkl', 'stroke_train_preprocessed.csv', 'stroke_test_preprocessed.csv']:
        need_file(f, 'Hãy chạy Tien_Xu_Ly.py trước!')

    # 1) Dựng lại tập train/test GỐC, đúng từng bước như Tien_Xu_Ly.py
    df = pd.read_csv(path('healthcare-dataset-stroke-data.csv')).drop(columns=['id'])
    df = df[df['gender'] != 'Other']
    X, y = df.drop(columns=['stroke']), df['stroke']
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 2) Preprocessor đã fit ở task 3 (chỉ dùng để lấy cấu hình + tên cột)
    prep_data = joblib.load(path('preprocessor.pkl'))
    preprocessor, feature_names = prep_data['preprocessor'], prep_data['feature_names']
    cat_indices = [i for i, n in enumerate(feature_names) if n not in NUMERIC_FEATURES]

    # 3) Các file của task 3: train đã SMOTENC (để huấn luyện mô hình cuối) và test (để đánh giá)
    train_df = pd.read_csv(path('stroke_train_preprocessed.csv'))
    test_df = pd.read_csv(path('stroke_test_preprocessed.csv'))

    # 4) Kiểm tra: tập test dựng lại từ dữ liệu gốc phải TRÙNG file test của task 3
    same_x = np.allclose(preprocessor.transform(X_test_raw), test_df[feature_names].values)
    same_y = bool((y_test_raw.values == test_df['stroke'].values).all())
    if not (same_x and same_y):
        print('Lỗi: tập test dựng lại KHÔNG trùng stroke_test_preprocessed.csv. '
              'Có thể Tien_Xu_Ly.py đã đổi cách chia dữ liệu; hãy sửa phần chia dữ liệu trong script này cho khớp.')
        raise SystemExit(1)

    return {
        'X_train_raw': X_train_raw, 'y_train_raw': y_train_raw,
        'preprocessor': preprocessor, 'feature_names': feature_names, 'cat_indices': cat_indices,
        'X_train_smote': train_df[feature_names], 'y_train_smote': train_df['stroke'],
        'X_test': test_df[feature_names], 'y_test': test_df['stroke'],
    }


def make_pipeline(clf, d):
    """tiền xử lý -> SMOTENC -> mô hình. Trong cross-validation, mọi bước đều chỉ fit trên fold train."""
    return ImbPipeline([
        ('prep', clone(d['preprocessor'])),  # bản CHƯA fit của preprocessor task 3
        ('smote', SMOTENC(categorical_features=d['cat_indices'], random_state=RANDOM_STATE)),
        ('clf', clf),
    ])


# ============================== ĐÁNH GIÁ ==============================
def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        'Accuracy': accuracy_score(y, pred),
        'Precision': precision_score(y, pred, zero_division=0),
        'Recall': recall_score(y, pred),
        'F1': f1_score(y, pred),
        'ROC-AUC': roc_auc_score(y, proba),
        'PR-AUC': average_precision_score(y, proba),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
    }


def demo_leakage(baselines, d, cv):
    """Mục 0: so sánh điểm F1 khi CV sai (trên file đã SMOTENC) với CV đúng và với test thật."""
    print('\n' + '=' * 78)
    print('0. VÌ SAO KHÔNG CROSS-VALIDATE TRỰC TIẾP TRÊN stroke_train_preprocessed.csv?')
    print('   (mô hình dùng đúng siêu tham số baseline của task 3, F1 của lớp stroke)')
    print('=' * 78)
    rows = {}
    for name, model in baselines.items():
        wrong = cross_validate(clone(model), d['X_train_smote'], d['y_train_smote'],
                               cv=cv, scoring='f1', n_jobs=N_JOBS)['test_score'].mean()
        right = cross_validate(make_pipeline(clone(model), d), d['X_train_raw'], d['y_train_raw'],
                               cv=cv, scoring='f1', n_jobs=N_JOBS)['test_score'].mean()
        rows[name] = {
            'CV trên file đã SMOTENC (SAI)': wrong,
            'CV đúng (SMOTENC trong fold)': right,
            'F1 trên test thật': f1_score(d['y_test'], model.predict(d['X_test'])),
        }
    print(pd.DataFrame(rows).T.round(3).to_string())
    print('=> CV sai cho điểm quá cao so với test; CV đúng sát thực tế hơn, nên dùng nó để chọn tham số.')


# ============================== TUNING ==============================
def run_search(name, search_cls, estimator, params, d, cv, **kwargs):
    search = search_cls(
        make_pipeline(estimator, d), params,
        scoring=SCORING, refit=REFIT_METRIC, cv=cv, n_jobs=N_JOBS,
        return_train_score=True, error_score='raise', verbose=1, **kwargs,
    )
    t0 = time.time()
    search.fit(d['X_train_raw'], d['y_train_raw'])
    print(f'-> {name}: xong sau {time.time() - t0:.0f}s')

    res = pd.DataFrame(search.cv_results_).sort_values(f'rank_test_{REFIT_METRIC}')
    best = res.iloc[0]
    print(f'   Tham số tốt nhất: { {k[5:]: v for k, v in search.best_params_.items()} }')
    print(f"   CV {REFIT_METRIC} = {best[f'mean_test_{REFIT_METRIC}']:.3f} ± {best[f'std_test_{REFIT_METRIC}']:.3f}"
          f"  (train {REFIT_METRIC} = {best[f'mean_train_{REFIT_METRIC}']:.3f})")
    top = res.head(5)[['params', f'mean_test_{REFIT_METRIC}', f'std_test_{REFIT_METRIC}']]
    print('   Top 5 cấu hình theo CV:')
    for _, r in top.iterrows():
        short = {k[5:]: v for k, v in r['params'].items()}
        print(f"     {r[f'mean_test_{REFIT_METRIC}']:.3f} ± {r[f'std_test_{REFIT_METRIC}']:.3f}  {short}")

    fname = 'cv_results_' + name.lower().replace(' ', '_') + '.csv'
    res.drop(columns=['params']).to_csv(os.path.join(OUT_DIR, fname), index=False)
    return search


def fit_tuned_model(search, d):
    """Huấn luyện lại bộ phân loại với tham số tốt nhất trên CHÍNH file train của task 3 (cùng dữ liệu với baseline)."""
    clf = clone(search.best_estimator_.named_steps['clf'])
    return clf.fit(d['X_train_smote'], d['y_train_smote'])


# ============================== BIỂU ĐỒ ==============================
def plot_confusion_matrices(models, d, file):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, (name, model) in zip(axes.ravel(), models.items()):
        ConfusionMatrixDisplay.from_predictions(
            d['y_test'], model.predict(d['X_test']), ax=ax, colorbar=False, values_format='d',
            display_labels=['Không stroke', 'Stroke'], cmap='Blues',
        )
        ax.set_title(name)
    fig.suptitle('Ma trận nhầm lẫn trên tập test')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, file), dpi=150)
    plt.close(fig)


def plot_pr_curves(models, d, file):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in models.items():
        PrecisionRecallDisplay.from_predictions(
            d['y_test'], model.predict_proba(d['X_test'])[:, 1], name=name, ax=ax
        )
    ax.axhline(d['y_test'].mean(), ls='--', c='gray', label=f"Đoán ngẫu nhiên ({d['y_test'].mean():.3f})")
    ax.set_title('Precision-Recall trên tập test')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, file), dpi=150)
    plt.close(fig)


def plot_dt_heatmap(search, file):
    """Điểm CV của Decision Tree theo max_depth và min_samples_leaf (tại criterion tốt nhất)."""
    crit = search.best_params_['clf__criterion']
    depths, leaves = DT_GRID['clf__max_depth'], DT_GRID['clf__min_samples_leaf']
    score = {(p['clf__criterion'], p['clf__max_depth'], p['clf__min_samples_leaf']): s
             for p, s in zip(search.cv_results_['params'], search.cv_results_[f'mean_test_{REFIT_METRIC}'])}
    z = np.array([[score[(crit, dep, lf)] for lf in leaves] for dep in depths])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(z, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(leaves)))
    ax.set_xticklabels([str(v) for v in leaves])
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([str(v) for v in depths])
    mid = (z.min() + z.max()) / 2
    for i in range(len(depths)):
        for j in range(len(leaves)):
            ax.text(j, i, f'{z[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='black' if z[i, j] > mid else 'white')
    ax.set_xlabel('min_samples_leaf')
    ax.set_ylabel('max_depth')
    ax.set_title(f'Decision Tree: {REFIT_METRIC} (CV) theo siêu tham số (criterion={crit})')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, file), dpi=150)
    plt.close(fig)


# ============================== MAIN ==============================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = load_data()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for f in BASELINE_FILES.values():
        need_file(f, 'Hãy chạy decision_tree_train.py và random_forest_train.py trước!')
    baselines = {name: joblib.load(path(f)) for name, f in BASELINE_FILES.items()}

    if RUN_LEAKAGE_DEMO:
        demo_leakage(baselines, d, cv)

    print('\n' + '=' * 78)
    print(f'1. TUNING (CV {N_SPLITS}-fold, chọn tham số theo {REFIT_METRIC})')
    print('=' * 78)
    search_dt = run_search('Decision Tree', GridSearchCV,
                           DecisionTreeClassifier(random_state=RANDOM_STATE), DT_GRID, d, cv)
    search_rf = run_search('Random Forest', RandomizedSearchCV,
                           RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1), RF_DIST, d, cv,
                           n_iter=N_ITER_RF, random_state=RANDOM_STATE)

    tuned = {'Decision Tree': fit_tuned_model(search_dt, d), 'Random Forest': fit_tuned_model(search_rf, d)}
    for name, model in tuned.items():
        joblib.dump(model, path(TUNED_FILES[name]), compress=3)

    # Thứ tự: DT baseline, DT tuned, RF baseline, RF tuned
    models = {}
    for name in ['Decision Tree', 'Random Forest']:
        models[f'{name} (baseline)'] = baselines[name]
        models[f'{name} (tuned)'] = tuned[name]

    print('\n' + '=' * 78)
    print('2. SO SÁNH BASELINE VÀ TUNED TRÊN TẬP TEST (chỉ đánh giá 1 lần)')
    print('=' * 78)
    rows = {}
    for name, model in models.items():
        rows[name] = evaluate(model, d['X_test'], d['y_test'])
        # F1 trên tập train (đã SMOTENC): gần 1.0 nghĩa là mô hình học thuộc dữ liệu train (overfit)
        rows[name]['F1 train'] = f1_score(d['y_train_smote'], model.predict(d['X_train_smote']))
    table = pd.DataFrame(rows).T
    table[['TN', 'FP', 'FN', 'TP']] = table[['TN', 'FP', 'FN', 'TP']].astype(int)
    print(table.round(3).to_string())
    print('(F1 train tính trên stroke_train_preprocessed.csv; càng gần 1.0 càng cho thấy mô hình học thuộc dữ liệu train)')
    table.round(4).to_csv(os.path.join(OUT_DIR, 'bang_so_sanh.csv'), encoding='utf-8-sig')

    best_params = {
        'metric_chon_tham_so': REFIT_METRIC,
        'Decision Tree': {k[5:]: v for k, v in search_dt.best_params_.items()},
        'Random Forest': {k[5:]: v for k, v in search_rf.best_params_.items()},
    }
    with open(os.path.join(OUT_DIR, 'best_params.json'), 'w', encoding='utf-8') as fh:
        json.dump(best_params, fh, ensure_ascii=False, indent=2)

    plot_confusion_matrices(models, d, 'confusion_matrix.png')
    plot_pr_curves(models, d, 'pr_curve.png')
    plot_dt_heatmap(search_dt, 'decision_tree_heatmap.png')

    print(f'\nĐã lưu bảng so sánh, biểu đồ, tham số tốt nhất vào: {OUT_DIR}')
    print('Đã lưu mô hình mới: ' + ', '.join(TUNED_FILES.values()))


if __name__ == '__main__':
    main()
