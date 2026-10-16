import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report

# k-fold testing
def k_foldProcess(
        randomStateCross=42,
        k=5, 
        modelCriterion='gini', 
        maxDepth=None, 
        maxLeaf=None,
        minSamplesSplit=2, 
        minSamplesLeaf=1, 
        classWeight=None, 
        ccpAlpha=0.0
    ):
    
    # k-Fold divide
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    features = df.drop(columns=['stroke'])
    target = df['stroke']
    k_foldDivide = StratifiedKFold(n_splits=k, shuffle=True, random_state=randomStateCross)

    # Start Performance evaluating
    # Define Metric to use in evaluation
    scoringMetrics = ['precision', 'recall', 'f1']

    # Decision Tree
    decisionTreeModel = model_decisionTree_declare(
        criterion=modelCriterion, 
        max_depth=maxDepth, 
        max_leaf_nodes=maxLeaf,
        min_samples_split=minSamplesSplit, 
        min_samples_leaf=minSamplesLeaf, 
        random_state=42,
        class_weight=classWeight, 
        ccp_alpha=ccpAlpha
    )
    decisionTreeResult = cross_validate(decisionTreeModel, features, target, cv=k_foldDivide, scoring=scoringMetrics, n_jobs=-1)

    # Random Forest
    randomForestModel = model_randomForest_declare(
        criterion=modelCriterion,
        max_depth=maxDepth, 
        max_leaf_nodes=maxLeaf,
        min_samples_split=minSamplesSplit, 
        min_samples_leaf=minSamplesLeaf,
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

# Model Evaluation
def modelEvaluation():
    randomStateCross1 = [0, 42, 100]
    k2 = [3, 5, 10]  
    modelCriterion3 = ['gini', 'entropy', 'log_loss'] 
    maxDepth4 = [None, 10, 20] 
    maxLeaf4 = [None, 20, 50]
    minSamplesSplit5 = [2, 8, 20] 
    minSamplesLeaf5 = [1, 5, 10] 
    classWeight6 = [None, 'balanced'] 
    ccpAlpha6 = [0.0, 0.0001, 0.001, 0.01, 0.5]

    print("Model Evaluation")
    for randomStateCross in randomStateCross1:
        k_foldProcess(randomStateCross=randomStateCross)

    for k in k2:
        k_foldProcess(k=k)

    for modelCriterion in modelCriterion3:
        k_foldProcess(modelCriterion=modelCriterion)

    for maxDepth, maxLeaf in zip(maxDepth4, maxLeaf4):
        k_foldProcess(maxDepth=maxDepth, maxLeaf=maxLeaf)

    for minSamplesSplit, minSamplesLeaf in zip(minSamplesSplit5, minSamplesLeaf5):
        k_foldProcess(minSamplesSplit=minSamplesSplit, minSamplesLeaf=minSamplesLeaf) 

    for classWeight, ccpAlpha in zip(classWeight6, ccpAlpha6):
        k_foldProcess(classWeight=classWeight, ccpAlpha=ccpAlpha)

if __name__ == "__main__":
    modelEvaluation()