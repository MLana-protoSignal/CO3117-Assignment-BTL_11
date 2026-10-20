"""
TASK 5 - TỐI ƯU THAM SỐ (HYPERPARAMETER TUNING) CHO DECISION TREE VÀ RANDOM FOREST
==================================================================================
Chạy (sau khi đã chạy Tien_Xu_Ly.py để có preprocessor.pkl):

    python hyperparameter_tuning.py

Script cần trong cùng thư mục: healthcare-dataset-stroke-data.csv, preprocessor.pkl,
decision_tree_train.py và random_forest_train.py (chỉ để lấy hàm khai báo mô hình của task 3).

Script làm các việc sau:
  0. (Tùy chọn) Minh họa vì sao KHÔNG được cross-validation trực tiếp trên dữ liệu đã SMOTENC.
  1. Tuning Decision Tree (GridSearchCV) và Random Forest (RandomizedSearchCV).
  2. So sánh Baseline (mô hình task 3) với Tuned trên CÙNG tập test.
  3. Lưu bảng so sánh, biểu đồ, tham số tốt nhất vào thư mục KetQuaTuning/
     và lưu 2 mô hình mới: decision_tree_tuned_model.pkl, random_forest_tuned_model.pkl.

VÌ SAO SCRIPT TỰ CHIA DỮ LIỆU TỪ FILE GỐC, KHÔNG DÙNG stroke_preprocessed.csv?
-----------------------------------------------------------------------------------
stroke_preprocessed.csv do Tien_Xu_Ly.py tạo ra đã bị SMOTENC trên TOÀN BỘ dữ liệu (và bộ tiền xử lý
cũng fit trên toàn bộ dữ liệu), nên không còn tập test thật nào để đánh giá: mỗi ca stroke "ảo" được
sinh ra từ các ca stroke thật nằm gần nó, và cả hai nằm lẫn trong mọi lần chia. Chia file này ra
train/validation/test thì mô hình chỉ cần nhớ là đoán đúng => điểm cao ảo => chọn sai tham số.
Cách làm đúng (dùng ở đây):
  - Chia train/test (80/20, stratify) từ file gốc TRƯỚC khi làm bất cứ việc gì khác.
  - Đặt  tiền xử lý -> SMOTENC -> mô hình  vào một imblearn Pipeline, để mỗi fold chỉ fit/SMOTE trên
    phần train của fold đó; phần validation giữ nguyên phân phối thật (~5% stroke).
  - Tập test không đụng tới cho đến bước so sánh cuối cùng.
  - Baseline được huấn luyện lại ngay trong script trên CÙNG tập train với mô hình tuned (bằng hàm
    khai báo của task 3), nên so sánh baseline với tuned là công bằng.
"""

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

try:  # hàm khai báo mô hình baseline của task 3 (2 file này phải nằm cùng thư mục với script)
    from decision_tree_train import model_decisionTree_declare
    from random_forest_train import model_randomForest_declare
except ImportError as err:
    raise SystemExit(f'Lỗi: không import được file train của task 3 ({err}). '
                     'Hãy đặt decision_tree_train.py và random_forest_train.py cùng thư mục với script này.')

# ============================== CẤU HÌNH ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'KetQuaTuning')

RANDOM_STATE = 42        # dùng cho mọi bước ngẫu nhiên (chia dữ liệu, SMOTENC, CV, mô hình)
N_SPLITS = 5             # số fold của cross-validation
N_ITER_RF = 30           # số cấu hình Random Forest được thử (RandomizedSearchCV)
N_JOBS = 1               # 1 = chạy tuần tự, ít tốn RAM (đặt -1 để dùng hết nhân CPU nếu máy khỏe)
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
DT_GRID = {  # 2 x 8 x 5 x 2 = 160 cấu hình -> đủ nhỏ để duyệt hết bằng GridSearchCV
    'clf__criterion': ['gini', 'entropy'],
    'clf__max_depth': [3, 4, 5, 6, 8, 10, 15, None],
    'clf__min_samples_leaf': [1, 5, 10, 20, 50],
    'clf__max_features': ['sqrt', None],  # 'sqrt' là mặc định trong model_decisionTree_declare của task 3
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
# Tham số baseline (task 3), truyền vào model_decisionTree_declare / model_randomForest_declare.
# Tham số không ghi ở đây lấy mặc định của hai hàm đó. Hãy đổi cho khớp với tham số nhóm chốt ở task 3.
BASELINE_PARAMS = {
    'Decision Tree': {'max_depth': 5},   # giống main.py: model_decisionTree_declare(max_depth=5)
    'Random Forest': {},                 # giống main.py: model_randomForest_declare()
}
SHOW_PARAMS = {  # tham số được in ra và ghi vào best_params.json để so sánh baseline với tuned
    'Decision Tree': ['criterion', 'max_depth', 'min_samples_leaf', 'max_features'],
    'Random Forest': ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                      'min_samples_leaf', 'max_features'],
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
    need_file('preprocessor.pkl', 'Hãy chạy Tien_Xu_Ly.py trước!')

    # 1) Làm sạch giống Tien_Xu_Ly.py, rồi chia train/test GỐC (80/20) TRƯỚC mọi bước xử lý khác
    df = pd.read_csv(path('healthcare-dataset-stroke-data.csv')).drop(columns=['id'])
    df = df[df['gender'] != 'Other']
    X, y = df.drop(columns=['stroke']), df['stroke']
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 2) Cấu hình tiền xử lý lấy từ preprocessor.pkl của Tien_Xu_Ly.py (clone = bản CHƯA fit) + tên cột
    prep_data = joblib.load(path('preprocessor.pkl'))
    preprocessor, feature_names = prep_data['preprocessor'], prep_data['feature_names']
    cat_indices = [i for i, n in enumerate(feature_names) if n not in NUMERIC_FEATURES]

    # 3) Fit tiền xử lý và SMOTENC CHỈ trên tập train; tập test chỉ transform, không SMOTE
    prep = clone(preprocessor).fit(X_train_raw)
    X_train_sm, y_train_sm = SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE).fit_resample(
        prep.transform(X_train_raw), y_train_raw
    )

    return {
        'X_train_raw': X_train_raw, 'y_train_raw': y_train_raw,
        'preprocessor': preprocessor, 'feature_names': feature_names, 'cat_indices': cat_indices,
        # train đã tiền xử lý + SMOTENC: dùng để huấn luyện baseline và mô hình tuned cuối cùng
        'X_train_smote': pd.DataFrame(X_train_sm, columns=feature_names),
        'y_train_smote': pd.Series(np.asarray(y_train_sm), name='stroke'),
        # test đã tiền xử lý (giữ nguyên phân phối thật): chỉ dùng để đánh giá cuối
        'X_test': pd.DataFrame(prep.transform(X_test_raw), columns=feature_names),
        'y_test': y_test_raw.reset_index(drop=True),
    }


def build_baselines(d):
    """Baseline task 3: khai báo bằng hàm của nhóm, huấn luyện trên CÙNG tập train (đã SMOTENC) với mô hình tuned."""
    baselines = {
        'Decision Tree': model_decisionTree_declare(**BASELINE_PARAMS['Decision Tree']),
        'Random Forest': model_randomForest_declare(n_jobs=1, **BASELINE_PARAMS['Random Forest']),
    }
    for model in baselines.values():
        model.fit(d['X_train_smote'], d['y_train_smote'])
    return baselines


def show_params(model, name):
    return {k: model.get_params()[k] for k in SHOW_PARAMS[name]}


def make_pipeline(clf, d):
    """tiền xử lý -> SMOTENC -> mô hình. Trong cross-validation, mọi bước đều chỉ fit trên fold train."""
    return ImbPipeline([
        ('prep', clone(d['preprocessor'])),  # bản CHƯA fit của preprocessor
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
    print('0. VÌ SAO KHÔNG CROSS-VALIDATE TRỰC TIẾP TRÊN DỮ LIỆU ĐÃ SMOTENC?')
    print('   (mô hình dùng đúng siêu tham số baseline của task 3, F1 của lớp stroke)')
    print('=' * 78)
    rows = {}
    for name, model in baselines.items():
        wrong = cross_validate(clone(model), d['X_train_smote'], d['y_train_smote'],
                               cv=cv, scoring='f1', n_jobs=N_JOBS)['test_score'].mean()
        right = cross_validate(make_pipeline(clone(model), d), d['X_train_raw'], d['y_train_raw'],
                               cv=cv, scoring='f1', n_jobs=N_JOBS)['test_score'].mean()
        rows[name] = {
            'CV trên dữ liệu đã SMOTENC (SAI)': wrong,
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
    """Huấn luyện lại bộ phân loại với tham số tốt nhất trên CHÍNH tập train đã SMOTENC (cùng dữ liệu với baseline)."""
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
    """Điểm CV của Decision Tree theo max_depth và min_samples_leaf (tại criterion và max_features tốt nhất)."""
    crit = search.best_params_['clf__criterion']
    mfeat = search.best_params_['clf__max_features']
    depths, leaves = DT_GRID['clf__max_depth'], DT_GRID['clf__min_samples_leaf']
    score = {(p['clf__criterion'], p['clf__max_features'], p['clf__max_depth'], p['clf__min_samples_leaf']): s
             for p, s in zip(search.cv_results_['params'], search.cv_results_[f'mean_test_{REFIT_METRIC}'])}
    z = np.array([[score[(crit, mfeat, dep, lf)] for lf in leaves] for dep in depths])

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
    ax.set_title(f'Decision Tree: {REFIT_METRIC} (CV) theo siêu tham số\n(criterion={crit}, max_features={mfeat})')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, file), dpi=150)
    plt.close(fig)


# ============================== MAIN ==============================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = load_data()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    baselines = build_baselines(d)
    print('Baseline (task 3), huấn luyện trên tập train đã SMOTENC:')
    for name, model in baselines.items():
        print(f'  {name}: {show_params(model, name)}')

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
    print('(F1 train tính trên tập train đã SMOTENC; càng gần 1.0 càng cho thấy mô hình học thuộc dữ liệu train)')
    table.round(4).to_csv(os.path.join(OUT_DIR, 'bang_so_sanh.csv'), encoding='utf-8-sig')

    best_params = {
        'metric_chon_tham_so': REFIT_METRIC,
        'baseline': {name: show_params(model, name) for name, model in baselines.items()},
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
