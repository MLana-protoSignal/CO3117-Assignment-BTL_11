import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def model_randomForest_declare(
    n_estimators=200,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features="sqrt",
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    random_state=42,
    n_jobs=-1,
    class_weight=None,
    ccp_alpha=0.0,
    monotonic_cst=None,
):
    # Declare and return a model directly from the passed-in parameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        monotonic_cst=monotonic_cst,
    )
    return model

def train_randomForest_model(model_input):
    # 1. Ask the user to enter the path to the training file (default = 'stroke_preprocessed.csv')
    path_input = input("Enter the path to the training file (default = 'stroke_preprocessed.csv'): ").strip()
    train_path = path_input if path_input else 'stroke_preprocessed.csv'
    
    # 2. Read data from path
    df_train = pd.read_csv(train_path)
    target_col = 'stroke'
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    # 3. Train model
    model_input.fit(X_train, y_train)
    
    # 4. Ask for the filename to save the model (default = 'randomForest_stroke_model.pkl')
    model_name_input = input("Enter filename to save the model (default = 'random_forest_stroke_model.pkl'): ").strip()
    if not model_name_input:
        save_filename = 'random_forest_stroke_model.pkl'
    else:
        # Automatically add the .pkl if the user does not enter it
        save_filename = model_name_input if model_name_input.endswith('.pkl') else f"{model_name_input}.pkl"

    # 5. Save model
    joblib.dump(model_input, save_filename)
    print(f"-> Model successfully saved to '{save_filename}'!")
    
    return model_input