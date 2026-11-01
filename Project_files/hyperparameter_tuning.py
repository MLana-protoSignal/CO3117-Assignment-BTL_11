"""
TASK 5 - HYPERPARAMETER TUNING FOR DECISION TREE AND RANDOM FOREST
==================================================================
Run (after Tien_Xu_Ly.py has been run to produce preprocessor.pkl):

    python hyperparameter_tuning.py

The script needs these files in the same folder: healthcare-dataset-stroke-data.csv, preprocessor.pkl,
decision_tree_train.py and random_forest_train.py (only used to get the Task 3 model declaration functions).

The script does the following:
  0. (Optional) Shows why cross-validation must NOT be run directly on SMOTENC-oversampled data.
  1. Tunes Decision Tree (GridSearchCV) and Random Forest (RandomizedSearchCV).
  2. Compares the Baseline (Task 3 model) with the Tuned model on the SAME test set.
  3. Saves the comparison table, plots and best parameters into the KetQuaTuning/ folder
     and saves the 2 new models: decision_tree_tuned_model.pkl, random_forest_tuned_model.pkl.

WHY DOES THE SCRIPT SPLIT THE DATA ITSELF FROM THE RAW FILE INSTEAD OF USING stroke_preprocessed.csv?
-----------------------------------------------------------------------------------------------------
stroke_preprocessed.csv, created by Tien_Xu_Ly.py, was SMOTENC-oversampled on the WHOLE dataset (and the
preprocessor was also fitted on the whole dataset), so no real test set is left to evaluate on: every
"synthetic" stroke case is generated from nearby real stroke cases, and both are mixed into every split.
If this file is split into train/validation/test, the model only has to memorize to predict correctly
=> inflated scores => wrong hyperparameters.
The correct approach (used here):
  - Split train/test (80/20, stratified) from the raw file BEFORE doing anything else.
  - Put  preprocessing -> SMOTENC -> model  into one imblearn Pipeline, so each fold only fits/oversamples
    on that fold's training part; the validation part keeps the real distribution (~5% stroke).
  - The test set is not touched until the final comparison step.
  - The baseline is retrained inside this script on the SAME training set as the tuned models (using the
    Task 3 declaration functions), so the comparison between baseline and tuned is fair.
"""

import json
import os
import time

import joblib
import matplotlib

matplotlib.use('Agg')  # only save images to file, do not open a window
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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
def model_decisionTree_declare(criterion='gini', max_depth=5, min_samples_leaf=10,
                               max_features=None, random_state=42):
    return DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  max_features=max_features, random_state=random_state)


def model_randomForest_declare(n_estimators=200, criterion='entropy', max_depth=None,
                               min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                               random_state=42, n_jobs=1):
    return RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                  max_depth=max_depth, min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf, max_features=max_features,
                                  random_state=random_state, n_jobs=n_jobs)

# ============================== CONFIGURATION ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'KetQuaTuning')

RANDOM_STATE = 42        # used for every random step (data split, SMOTENC, CV, models)
N_SPLITS = 5             # number of cross-validation folds
N_ITER_RF = 30           # number of Random Forest configurations to try (RandomizedSearchCV)
N_JOBS = 1               # 1 = run sequentially, low RAM usage (set -1 to use all CPU cores if the machine is strong)
RUN_LEAKAGE_DEMO = True  # print the data leakage demonstration table (section 0)

# Metric used to SELECT the best parameters. The data is imbalanced (~5% stroke) so accuracy is not used.
# F1 of the stroke class balances missed cases (recall) against false alarms (precision).
# Can be changed to 'average_precision' (PR-AUC) or 'recall'.
REFIT_METRIC = 'f1'
SCORING = {  # metrics recorded in every cross-validation run
    'f1': 'f1',
    'recall': 'recall',
    'precision': 'precision',
    'average_precision': 'average_precision',
    'roc_auc': 'roc_auc',
}

# class_weight is not tuned because SMOTENC already balances the classes 50/50 in every training fold.
DT_GRID = {  # 2 x 8 x 5 x 2 = 160 configurations -> small enough to search exhaustively with GridSearchCV
    'clf__criterion': ['gini', 'entropy'],
    'clf__max_depth': [3, 4, 5, 6, 8, 10, 15, None],
    'clf__min_samples_leaf': [1, 5, 10, 20, 50],
    'clf__max_features': ['sqrt', None],  # 'sqrt' is the default in Task 3's model_decisionTree_declare
}
RF_DIST = {  # 2160 configurations -> too many to search exhaustively, only sample N_ITER_RF at random
    'clf__n_estimators': [100, 200, 300, 500],
    'clf__criterion': ['gini', 'entropy'],
    'clf__max_depth': [5, 8, 10, 15, 20, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4, 8, 16],
    'clf__max_features': ['sqrt', 'log2', 0.5],
}

NUMERIC_FEATURES = ['avg_glucose_level', 'bmi', 'age']  # numeric columns in Tien_Xu_Ly.py (the rest are 0/1)
# Baseline (Task 3) parameters, passed to model_decisionTree_declare / model_randomForest_declare.
# Parameters not listed here take the defaults of those two functions. Change them to match what the group fixed in Task 3.
BASELINE_PARAMS = {
    'Decision Tree': {'max_depth': 5},   # same as main.py: model_decisionTree_declare(max_depth=5)
    'Random Forest': {},                 # same as main.py: model_randomForest_declare()
}
SHOW_PARAMS = {  # parameters printed and written to best_params.json to compare baseline with tuned
    'Decision Tree': ['criterion', 'max_depth', 'min_samples_leaf', 'max_features'],
    'Random Forest': ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                      'min_samples_leaf', 'max_features'],
}
VI_NAMES = {  # Vietnamese names used on the figures
    'Decision Tree (baseline)': 'Cây quyết định (cơ sở)',
    'Decision Tree (tuned)': 'Cây quyết định (đã tối ưu)',
    'Random Forest (baseline)': 'Rừng ngẫu nhiên (cơ sở)',
    'Random Forest (tuned)': 'Rừng ngẫu nhiên (đã tối ưu)',
}
TUNED_FILES = {
    'Decision Tree': 'decision_tree_tuned_model.pkl',
    'Random Forest': 'random_forest_tuned_model.pkl',
}


def vn(x, nd=2):
    """Decimal number with a comma separator (for labels on the figures)."""
    return f'{x:.{nd}f}'.replace('.', ',')


def comma_axis(axis):
    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))


def path(name):
    return os.path.join(BASE_DIR, name)


def need_file(filename, hint):
    if not os.path.exists(path(filename)):
        print(f"Error: '{filename}' not found. {hint}")
        raise SystemExit(1)


# ============================== DATA ==============================
def load_data():
    need_file('healthcare-dataset-stroke-data.csv', 'Please put the raw data file in the same folder as the script.')
    need_file('preprocessor.pkl', 'Please run Tien_Xu_Ly.py first!')

    # 1) Clean the data like Tien_Xu_Ly.py, then make the ORIGINAL train/test split (80/20) BEFORE any other step
    df = pd.read_csv(path('healthcare-dataset-stroke-data.csv')).drop(columns=['id'])
    df = df[df['gender'] != 'Other']
    X, y = df.drop(columns=['stroke']), df['stroke']
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 2) Preprocessing configuration comes from preprocessor.pkl of Tien_Xu_Ly.py (clone = UNFITTED copy) + column names
    prep_data = joblib.load(path('preprocessor.pkl'))
    preprocessor, feature_names = prep_data['preprocessor'], prep_data['feature_names']
    cat_indices = [i for i, n in enumerate(feature_names) if n not in NUMERIC_FEATURES]

    # 3) Fit preprocessing and SMOTENC ONLY on the train set; the test set is only transformed, no SMOTE
    prep = clone(preprocessor).fit(X_train_raw)
    X_train_sm, y_train_sm = SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE).fit_resample(
        prep.transform(X_train_raw), y_train_raw
    )

    return {
        'X_train_raw': X_train_raw, 'y_train_raw': y_train_raw,
        'preprocessor': preprocessor, 'feature_names': feature_names, 'cat_indices': cat_indices,
        # train set after preprocessing + SMOTENC: used to train the baseline and the final tuned models
        'X_train_smote': pd.DataFrame(X_train_sm, columns=feature_names),
        'y_train_smote': pd.Series(np.asarray(y_train_sm), name='stroke'),
        # preprocessed test set (real distribution kept): only used for the final evaluation
        'X_test': pd.DataFrame(prep.transform(X_test_raw), columns=feature_names),
        'y_test': y_test_raw.reset_index(drop=True),
    }


def build_baselines(d):
    """Task 3 baseline: declared with the group's functions, trained on the SAME (SMOTENC) train set as the tuned models."""
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
    """preprocessing -> SMOTENC -> model. In cross-validation, every step is fitted only on the training fold."""
    return ImbPipeline([
        ('prep', clone(d['preprocessor'])),  # UNFITTED copy of the preprocessor
        ('smote', SMOTENC(categorical_features=d['cat_indices'], random_state=RANDOM_STATE)),
        ('clf', clf),
    ])


# ============================== EVALUATION ==============================
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
    """Section 0: compare the F1 score of wrong CV (on SMOTENC data) with correct CV and with the real test set."""
    print('\n' + '=' * 78)
    print('0. WHY NOT CROSS-VALIDATE DIRECTLY ON SMOTENC-OVERSAMPLED DATA?')
    print('   (models use the exact Task 3 baseline hyperparameters, F1 of the stroke class)')
    print('=' * 78)
    rows = {}
    for name, model in baselines.items():
        wrong = cross_validate(clone(model), d['X_train_smote'], d['y_train_smote'],
                               cv=cv, scoring='f1', n_jobs=N_JOBS)['test_score'].mean()
        right = cross_validate(make_pipeline(clone(model), d), d['X_train_raw'], d['y_train_raw'],
                               cv=cv, scoring='f1', n_jobs=N_JOBS)['test_score'].mean()
        rows[name] = {
            'CV on SMOTENC data (WRONG)': wrong,
            'Correct CV (SMOTENC inside fold)': right,
            'F1 on real test set': f1_score(d['y_test'], model.predict(d['X_test'])),
        }
    print(pd.DataFrame(rows).T.round(3).to_string())
    print('=> Wrong CV gives scores far too high compared with the test set; correct CV is closer to reality, so it is used to select parameters.')


# ============================== TUNING ==============================
def run_search(name, search_cls, estimator, params, d, cv, **kwargs):
    search = search_cls(
        make_pipeline(estimator, d), params,
        scoring=SCORING, refit=REFIT_METRIC, cv=cv, n_jobs=N_JOBS,
        return_train_score=True, error_score='raise', verbose=1, **kwargs,
    )
    t0 = time.time()
    search.fit(d['X_train_raw'], d['y_train_raw'])
    print(f'-> {name}: finished after {time.time() - t0:.0f}s')

    res = pd.DataFrame(search.cv_results_).sort_values(f'rank_test_{REFIT_METRIC}')
    best = res.iloc[0]
    print(f'   Best parameters: { {k[5:]: v for k, v in search.best_params_.items()} }')
    print(f"   CV {REFIT_METRIC} = {best[f'mean_test_{REFIT_METRIC}']:.3f} ± {best[f'std_test_{REFIT_METRIC}']:.3f}"
          f"  (train {REFIT_METRIC} = {best[f'mean_train_{REFIT_METRIC}']:.3f})")
    top = res.head(5)[['params', f'mean_test_{REFIT_METRIC}', f'std_test_{REFIT_METRIC}']]
    print('   Top 5 configurations by CV:')
    for _, r in top.iterrows():
        short = {k[5:]: v for k, v in r['params'].items()}
        print(f"     {r[f'mean_test_{REFIT_METRIC}']:.3f} ± {r[f'std_test_{REFIT_METRIC}']:.3f}  {short}")

    fname = 'cv_results_' + name.lower().replace(' ', '_') + '.csv'
    res.drop(columns=['params']).to_csv(os.path.join(OUT_DIR, fname), index=False)
    return search


def fit_tuned_model(search, d):
    """Retrain the classifier with the best parameters on the very same SMOTENC train set (same data as the baseline)."""
    clf = clone(search.best_estimator_.named_steps['clf'])
    return clf.fit(d['X_train_smote'], d['y_train_smote'])


# ============================== PLOTS ==============================
def plot_confusion_matrices(models, d, file):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, (name, model) in zip(axes.ravel(), models.items()):
        ConfusionMatrixDisplay.from_predictions(
            d['y_test'], model.predict(d['X_test']), ax=ax, colorbar=False, values_format='d',
            display_labels=['Không đột quỵ', 'Đột quỵ'], cmap='Blues',
        )
        ax.set_title(VI_NAMES.get(name, name))
        ax.set_xlabel('Nhãn dự đoán')
        ax.set_ylabel('Nhãn thật')
    fig.suptitle('Ma trận nhầm lẫn trên tập kiểm tra')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, file), dpi=150)
    plt.close(fig)


def plot_pr_curves(models, d, file):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in models.items():
        proba = model.predict_proba(d['X_test'])[:, 1]
        precision, recall, _ = precision_recall_curve(d['y_test'], proba)
        ap = average_precision_score(d['y_test'], proba)
        ax.step(recall, precision, where='post', label=f"{VI_NAMES.get(name, name)} (PR-AUC = {vn(ap)})")
    rate = d['y_test'].mean()
    ax.axhline(rate, ls='--', c='gray', label=f'Đoán ngẫu nhiên ({vn(rate, 3)})')
    ax.set_xlabel('Độ nhạy (recall)')
    ax.set_ylabel('Độ chính xác dương (precision)')
    comma_axis(ax.xaxis)
    comma_axis(ax.yaxis)
    ax.set_title('Đường Precision-Recall trên tập kiểm tra')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, file), dpi=150)
    plt.close(fig)


def plot_dt_heatmap(search, file):
    """CV scores of the Decision Tree by max_depth and min_samples_leaf (at the best criterion and max_features)."""
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
            ax.text(j, i, vn(z[i, j]), ha='center', va='center', fontsize=8,
                    color='black' if z[i, j] > mid else 'white')
    ax.set_xlabel('min_samples_leaf')
    ax.set_ylabel('max_depth')
    ax.set_title(f'Cây quyết định: điểm {REFIT_METRIC.upper()} (kiểm chứng chéo) theo siêu tham số\n(criterion={crit}, max_features={mfeat})')
    cb = fig.colorbar(im, ax=ax)
    comma_axis(cb.ax.yaxis)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, file), dpi=150)
    plt.close(fig)


# ============================== MAIN ==============================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = load_data()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    baselines = build_baselines(d)
    print('Baseline (Task 3), trained on the SMOTENC-oversampled train set:')
    for name, model in baselines.items():
        print(f'  {name}: {show_params(model, name)}')

    if RUN_LEAKAGE_DEMO:
        demo_leakage(baselines, d, cv)

    print('\n' + '=' * 78)
    print(f'1. TUNING ({N_SPLITS}-fold CV, parameters selected by {REFIT_METRIC})')
    print('=' * 78)
    search_dt = run_search('Decision Tree', GridSearchCV,
                           DecisionTreeClassifier(random_state=RANDOM_STATE), DT_GRID, d, cv)
    search_rf = run_search('Random Forest', RandomizedSearchCV,
                           RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1), RF_DIST, d, cv,
                           n_iter=N_ITER_RF, random_state=RANDOM_STATE)

    tuned = {'Decision Tree': fit_tuned_model(search_dt, d), 'Random Forest': fit_tuned_model(search_rf, d)}
    for name, model in tuned.items():
        joblib.dump(model, path(TUNED_FILES[name]), compress=3)

    # Order: DT baseline, DT tuned, RF baseline, RF tuned
    models = {}
    for name in ['Decision Tree', 'Random Forest']:
        models[f'{name} (baseline)'] = baselines[name]
        models[f'{name} (tuned)'] = tuned[name]

    print('\n' + '=' * 78)
    print('2. BASELINE VS TUNED ON THE TEST SET (evaluated only once)')
    print('=' * 78)
    rows = {}
    for name, model in models.items():
        rows[name] = evaluate(model, d['X_test'], d['y_test'])
        # F1 on the train set (SMOTENC-oversampled): close to 1.0 means the model memorized the training data (overfit)
        rows[name]['F1 train'] = f1_score(d['y_train_smote'], model.predict(d['X_train_smote']))
    table = pd.DataFrame(rows).T
    table[['TN', 'FP', 'FN', 'TP']] = table[['TN', 'FP', 'FN', 'TP']].astype(int)
    print(table.round(3).to_string())
    print('(F1 train is computed on the SMOTENC-oversampled train set; the closer to 1.0, the more the model has memorized the training data)')
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

    print(f'\nSaved the comparison table, plots and best parameters to: {OUT_DIR}')
    print('Saved new models: ' + ', '.join(TUNED_FILES.values()))


if __name__ == '__main__':
    main()
