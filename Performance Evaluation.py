import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report

# k-fold testing
def k_foldProcess(
        k=5, 
        modelCriterion='gini', 
        maxDepth=None, 
        minSamplesSplit=2, 
        minSamplesLeaf=1, 
        maxLeaf=None,
        classWeight=None, 
        ccpAlpha=0.0
    ):
    
    # k-Fold divide
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    features = df.drop(columns=['stroke'])
    target = df['stroke']
    k_foldDivide = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Start Performance evaluating
    # Define Metric to use in evaluation
    scoringMetrics = ['precision', 'recall', 'f1']

    # Decision Tree
    decisionTreeModel = model_decisionTree_declare(
        criterion=modelCriterion, 
        max_depth=maxDepth, 
        min_samples_split=minSamplesSplit, 
        min_samples_leaf=minSamplesLeaf, 
        max_leaf_nodes=maxLeaf,
        random_state=42,
        class_weight=classWeight, 
        ccp_alpha=ccpAlpha
    )
    decisionTreeResult = cross_validate(decisionTreeModel, features, target, cv=k_foldDivide, scoring=scoringMetrics, n_jobs=-1)

    # Random Forest
    randomForestModel = model_randomForest_declare(
        criterion=modelCriterion,
        max_depth=maxDepth, 
        min_samples_split=minSamplesSplit, 
        min_samples_leaf=minSamplesLeaf,
        max_leaf_nodes=maxLeaf, 
        random_state=42,
        class_weight=classWeight, 
        ccp_alpha=ccpAlpha
    )
    randomForestResult = cross_validate(randomForestModel, features, target, cv=k_foldDivide, scoring=scoringMetrics, n_jobs=-1)

    # Comparison Data
    comparisonResult = pd.DataFrame({
        'Metrics': [
            'Precision Mean', 
            'Precision Variance', 
            'Recall Mean', 
            'Recall Variance', 
            'F1-Score Mean', 
            'F1-Score Variance'
        ],
        'Decision tree': [
            decisionTreeResult['test_precision'].mean(),
            decisionTreeResult['test_precision'].var(),
            decisionTreeResult['test_recall'].mean(),
            decisionTreeResult['test_recall'].var(),
            decisionTreeResult['test_f1'].mean(),
            decisionTreeResult['test_f1'].var()
        ],
        'Random Forest': [
            randomForestResult['test_precision'].mean(),
            randomForestResult['test_precision'].var(),
            randomForestResult['test_recall'].mean(),
            randomForestResult['test_recall'].var(),
            randomForestResult['test_f1'].mean(),
            randomForestResult['test_f1'].var()
        ]
    })
    print(comparisonResult.to_string(index=False))
    # Finish Performance evaluating 


if __name__ == "__main__":
    k = input("Number fold to divide for Performance Evaluation: ").strip()
    modelCriterion = input("Criterion: ").strip() 
    maxDepth = input("Maximum depth: ").strip() 
    minSamplesSplit = input("Minimum number of samples required to split an internal node: ").strip() 
    minSamplesLeaf = input("Minimum number of samples required to be at a leaf node: ").strip() 
    maxLeaf = input("Maximum number of leaf nodes: ").strip() 
    classWeight = input("Class Weight: ").strip()
    ccpAlpha = input("Complexity parameter used for Minimal Cost-Complexity Pruning: ").strip() 
    k_foldProcess(
        k=k, 
        modelCriterion=modelCriterion, 
        maxDepth=maxDepth, 
        minSamplesSplit=minSamplesSplit, 
        minSamplesLeaf=minSamplesLeaf, 
        maxLeaf=maxLeaf,
        classWeight=classWeight, 
        ccpAlpha=ccpAlpha, 
    )