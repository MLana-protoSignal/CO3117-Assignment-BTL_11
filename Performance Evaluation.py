import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
    
# k-fold testing
def k_foldProcess(k=5, numjob=-1):
    # k-Fold divide
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    features = df.drop(columns=['stroke'])
    target = df['stroke']
    k_foldDivide = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Start Performance evaluating
    # Define Metric to use in evaluation
    scoringMetrics = ['accuracy', 'precision', 'recall', 'f1', 'neg_log_loss']

    # Decision Tree
    decisionTreeModel = train_and_save_model() #chỉnh lại tên hàm này
    decisionTreeResult = cross_validate(decisionTreeModel, features, target, cv=k_foldDivide, scoring=scoringMetrics, n_jobs=numjob)

    # Random Forest
    randomForestModel = train_random_forest() #chỉnh lại tên hàm này
    randomForestResult = cross_validate(randomForestModel, features, target, cv=k_foldDivide, scoring=scoringMetrics, n_jobs=numjob)

    # Comparison Data
    comparisonResult = pd.DataFrame({
        'Metric': [
            'Accuracy Mean', 
            'Accuracy Variance', 
            'Precision Mean', 
            'Precision Variance', 
            'Recall Mean', 
            'Recall Variance', 
            'F1-Score Mean', 
            'F1-Score Variance', 
            'Log Loss Mean', 
            'Log Loss Variance',
            'Training Time Mean',
            'Training Time Variance'
        ],
        'Decision tree': [
            decisionTreeResult['test_accuracy'].mean(),
            decisionTreeResult['test_accuracy'].var(),
            decisionTreeResult['test_precision'].mean(),
            decisionTreeResult['test_precision'].var(),
            decisionTreeResult['test_recall'].mean(),
            decisionTreeResult['test_recall'].var(),
            decisionTreeResult['test_f1'].mean(),
            decisionTreeResult['test_f1'].var(),
            decisionTreeResult['test_neg_log_loss'].mean(),
            decisionTreeResult['test_neg_log_loss'].var(),
            decisionTreeResult['fit_time'].mean(),
            decisionTreeResult['fit_time'].var()
        ],
        'Random Forest': [
            randomForestResult['test_accuracy'].mean(),
            randomForestResult['test_accuracy'].var(),
            randomForestResult['test_precision'].mean(),
            randomForestResult['test_precision'].var(),
            randomForestResult['test_recall'].mean(),
            randomForestResult['test_recall'].var(),
            randomForestResult['test_f1'].mean(),
            randomForestResult['test_f1'].var(),
            randomForestResult['test_neg_log_loss'].mean(),
            randomForestResult['test_neg_log_loss'].var(),
            randomForestResult['fit_time'].mean(),
            randomForestResult['fit_time'].var()
        ]
    })
    print(comparisonResult.to_string(index=False))
    # Finish Performance evaluating 


if __name__ == "__main__":
    k2fold = input("Number fold to divide for Performance Evaluation: ").strip()
    jobs = input("Number of CPU processes running in parallel: ").strip()
    k_foldProcess(k=k2fold, numjob=jobs)