from random_forest_train import model_randomForest_declare
from decision_tree_train import model_decisionTree_declare
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

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
    # Start Preprocessing
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df = df.drop(columns=['id'], errors='ignore')
    df = df[df['gender'] != 'Other']
    df = df.dropna(subset=['stroke'])
    features = df.drop(columns=['stroke'])
    target = df['stroke']

    num_cols_skewed = ['avg_glucose_level', 'bmi']  
    num_cols_normal = ['age']  
    binary_cols = ['hypertension', 'heart_disease']  
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']  

    num_skewed_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log_transform', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', StandardScaler())
    ])
    num_normal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num_skewed', num_skewed_pipeline, num_cols_skewed),
        ('num_normal', num_normal_pipeline, num_cols_normal),
        ('binary', 'passthrough', binary_cols),  # Giữ nguyên giá trị 0/1 của 2 cột này
        ('cat', cat_pipeline, cat_cols)
    ])
    X_preprocessed = preprocessor.fit_transform(features)

    encoded_cat_cols = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols))
    feature_names = num_cols_skewed + num_cols_normal + binary_cols + encoded_cat_cols
    categorical_feature_names = binary_cols + encoded_cat_cols
    cat_indices = [feature_names.index(col) for col in categorical_feature_names]
    # End Preprocessing

    k_foldDivide = StratifiedKFold(n_splits=k, shuffle=True, random_state=randomStateCross)

    # Start Performance evaluating
    # Define Metric to use in evaluation
    scoringMetrics =  {
        'precision': make_scorer(
            precision_score,
            pos_label=1,
            zero_division=0
        ),

        'recall': make_scorer(
            recall_score,
            pos_label=1,
            zero_division=0
        ),

        'f1': make_scorer(
            f1_score,
            pos_label=1,
            zero_division=0
        )
    }

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
    decisionTreePipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTENC(categorical_features=cat_indices,random_state=42)),
        ('classifier', decisionTreeModel)
    ])
    decisionTreeResult = cross_validate(estimator=decisionTreePipeline,X=features,y=target,cv=k_foldDivide,scoring=scoringMetrics,n_jobs=-1)

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
    randomForestPipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTENC(categorical_features=cat_indices,random_state=42)),
        ('classifier', randomForestModel)
    ])
    randomForestResult = cross_validate(estimator=randomForestPipeline,X=features,y=target,cv=k_foldDivide,scoring=scoringMetrics,n_jobs=-1)

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
    maxDepth4 = [None, 5, 10, 20] 
    maxLeaf4 = [None, 10, 30, 50]
    minSamplesSplit5 = [2, 5, 10, 20] 
    minSamplesLeaf5 = [1, 2, 5, 10] 
    classWeight6 = [None, 'balanced'] 
    ccpAlpha6 = [0.0, 0.0001, 0.001, 0.01]

    print("\n================================ MODEL PERFORMANCE EVALUATION ================================\n")
    
    # SCENARIO 1: Change random_state of cross_validate() function
    print("SCENARIO 1: Change random_state of cross_validate() function\n")
    for randomStateCross in randomStateCross1:
        k_foldProcess(randomStateCross=randomStateCross)

    # SCENARIO 2: Change k of cross_validate() function
    print("\nSCENARIO 2: Change k of cross_validate() function\n")
    for k in k2:
        k_foldProcess(k=k)

    # SCENARIO 3: Change criterion of model_decisionTree_declare() and model_randomForest_declare()
    print("\nSCENARIO 3: Change criterion of model_decisionTree_declare() and model_randomForest_declare()\n")
    for modelCriterion in modelCriterion3:
        k_foldProcess(modelCriterion=modelCriterion)

    # SCENARIO 4: Change max_depth and max_leaf_nodes of model_decisionTree_declare() and model_randomForest_declare()
    print("\nSCENARIO 4: Change max_depth and max_leaf_nodes of model_decisionTree_declare() and model_randomForest_declare()\n")
    for maxDepth in maxDepth4:
        for maxLeaf in maxLeaf4:
            k_foldProcess(maxDepth=maxDepth,maxLeaf=maxLeaf)

    # SCENARIO 5: Change min_samples_split and min_samples_leaf of model_decisionTree_declare() and model_randomForest_declare()
    print("\nSCENARIO 5: Change min_samples_split and min_samples_leaf of model_decisionTree_declare() and model_randomForest_declare()\n")
    for minSamplesSplit in minSamplesSplit5:
        for minSamplesLeaf in minSamplesLeaf5:
            k_foldProcess(minSamplesSplit=minSamplesSplit,minSamplesLeaf=minSamplesLeaf) 

    # SCENARIO 6: Change class_weight and ccp_alpha of model_decisionTree_declare() and model_randomForest_declare()
    print("\nSCENARIO 6: Change class_weight and ccp_alpha of model_decisionTree_declare() and model_randomForest_declare()\n")
    for classWeight in classWeight6:
        for ccpAlpha in ccpAlpha6:
            k_foldProcess(classWeight=classWeight,ccpAlpha=ccpAlpha)

    print("\n==============================================================================================\n")

if __name__ == "__main__":
    modelEvaluation()