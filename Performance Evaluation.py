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

    print(f"----------------------------------------------------------------------------------------------")
    print(f"k-Fold Cross-Validation: random_state = {randomStateCross}")
    print(f"k-Fold Cross-Validation: shuffle = True")
    print(f"k-Fold Cross-Validation: k = {k}")
    print(f"Model: random_state = 42")
    print(f"Model: criterion = {modelCriterion}")
    print(f"Model: max_depth = {maxDepth}")
    print(f"Model: max_leaf_nodes = {maxLeaf}")
    print(f"Model: min_samples_split = {minSamplesSplit}")
    print(f"Model: min_samples_leaf = {minSamplesLeaf}")
    print(f"Model: class_weight = {classWeight}")
    print(f"Model: ccp_alpha = {ccpAlpha}")
    
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
        random_state=42,
        criterion=modelCriterion, 
        max_depth=maxDepth, 
        max_leaf_nodes=maxLeaf,
        min_samples_split=minSamplesSplit, 
        min_samples_leaf=minSamplesLeaf, 
        class_weight=classWeight, 
        ccp_alpha=ccpAlpha
    )
    decisionTreeResult = cross_validate(decisionTreeModel, features, target, cv=k_foldDivide, scoring=scoringMetrics, n_jobs=-1)

    # Random Forest
    randomForestModel = model_randomForest_declare(
        random_state=42,
        criterion=modelCriterion,
        max_depth=maxDepth, 
        max_leaf_nodes=maxLeaf,
        min_samples_split=minSamplesSplit, 
        min_samples_leaf=minSamplesLeaf,
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
    print(f"----------------------------------------------------------------------------------------------")
    # Finish Performance evaluating 

# Model Performance Evaluation
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

    print("================================ MODEL PERFORMANCE EVALUATION ================================")
    
    # SCENARIO 1: Change random_state of cross_validate() function
    print("SCENARIO 1:")
    for randomStateCross in randomStateCross1:
        k_foldProcess(randomStateCross=randomStateCross)

    # SCENARIO 2: Change k of cross_validate() function
    print("SCENARIO 2:")
    for k in k2:
        k_foldProcess(k=k)

    # SCENARIO 3: Change criterion of model_decisionTree_declare() and model_randomForest_declare()
    print("SCENARIO 3:")
    for modelCriterion in modelCriterion3:
        k_foldProcess(modelCriterion=modelCriterion)

    # SCENARIO 4: Change max_depth and max_leaf_nodes of model_decisionTree_declare() and model_randomForest_declare()
    print("SCENARIO 4:")
    for maxDepth, maxLeaf in zip(maxDepth4, maxLeaf4):
        k_foldProcess(maxDepth=maxDepth, maxLeaf=maxLeaf)

    # SCENARIO 5: Change min_samples_split and min_samples_leaf of model_decisionTree_declare() and model_randomForest_declare()
    print("SCENARIO 5:")
    for minSamplesSplit, minSamplesLeaf in zip(minSamplesSplit5, minSamplesLeaf5):
        k_foldProcess(minSamplesSplit=minSamplesSplit, minSamplesLeaf=minSamplesLeaf) 

    # SCENARIO 6: Change class_weight and ccp_alpha of model_decisionTree_declare() and model_randomForest_declare()
    print("SCENARIO 6:")
    for classWeight, ccpAlpha in zip(classWeight6, ccpAlpha6):
        k_foldProcess(classWeight=classWeight, ccpAlpha=ccpAlpha)

    print("==============================================================================================")

if __name__ == "__main__":
    modelEvaluation()