import os
import sys
import subprocess
from decision_tree_train import *
from random_forest_train import *
from performance_evaluation import *
from preprocess import *
REQUIRED_PACKAGES = [
    'pandas',
    'numpy',
    'scikit-learn',
    'imbalanced-learn',
    'joblib'
]

def install_and_import():
    # Automatically check for and install any missing libraries
    print("=" * 94)
    print(' CHECKING AND INSTALLING NECESSARY LIBRARIES...'.center(94))
    print("=" * 94)

    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"[*] Library '{package}' is not installed. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"[+] '{package}' is installed successfully.\n")
            except Exception as e:
                print(f"[!] Error install '{package}': {e}")
                sys.exit(1)

    print("[DONE] All necessary libraries are ready!\n")

def choose_process():
    # Ask user process to do
    print("-" * 94)
    print("Please choose your process:")
    print("  1. Prediction")
    print("  2. Performance Evaluation")
    print("  0. End")

    inputChoice = input("Enter your choice (1/2/0, default = 1): ").strip()
    if inputChoice == '2':
        choice = 2
    elif inputChoice == '0':
        choice = 0
    else:
        choice = 1
    print("-" * 94)

    return choice

def choose_algorithm_PE():
    # Ask user algorithm to do with the algorithm he/she chose
    print("-" * 94)
    print("Please choose your algorithm:")
    print("  1. Decision Tree")
    print("  2. Random Forest")
    print("  3. Both algorithm")
    print("  0. Return the last step")

    inputChoice = input("Enter your choice (1/2/3/0, default = 1): ").strip()
    if inputChoice == '2':
        choice = 2
    elif inputChoice == '3':
        choice = 3
    elif inputChoice == '0':
        choice = 0
    else:
        choice = 1
    print("-" * 94)

    return choice

def choose_algorithm_predict():
    # Ask user algorithm to do with the algorithm he/she chose
    print("-" * 94)
    print("Please choose your algorithm:")
    print("  1. Decision Tree")
    print("  2. Random Forest")
    print("  0. Return the last step")

    inputChoice = input("Enter your choice (1/2/0, default = 1): ").strip()
    if inputChoice == '2':
        choice = 2
    elif inputChoice == '0':
        choice = 0
    else:
        choice = 1
    print("-" * 94)

    return choice

def choose_model():
    # Ask user algorithm to do with the model he/she chose
    print("-" * 94)
    print("Please choose your model:")
    print("  1. Default model")
    print("  2. Input model")
    print("  0. Return the Main menu")

    inputChoice = input("Enter your choice (1/2/0, default = 1): ").strip()
    if inputChoice == '2':
        choice = 2
    elif inputChoice == '0':
        choice = 0
    else:
        choice = 1
    print("-" * 94)

    return choice

def run_predictor(script_name):
    # Run the predictor file that save in script_name
    if not os.path.exists(script_name):
        print(f"\n[!] Error: File '{script_name}' is not found in the current folder.")
        return

    print(f"\n---> Starting '{script_name}'...")
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except KeyboardInterrupt:
        print("\n[!] Stop the current program!")
    except Exception as e:
        print(f"\n[!] Error occurred when executing '{script_name}': {e}")

def main():
    # 1. Download and install missing libraries
    install_and_import()

    # 2. Program start
    print("=" * 94)
    print(' STROKE PREDICTION TOOL '.center(94))
    print("=" * 94)

    # Preprocessing
    path_input = input("\nEnter the path to the create preprocessed file (default = 'healthcare-dataset-stroke-data.csv'): ").strip()
    train_path = path_input if path_input else 'healthcare-dataset-stroke-data.csv'
    system_preprocessing(train_path)
    print(" ")

    # Models declaration
    # Add the finished model & Tuning
    decisionTreeModel = model_decisionTree_declare(
        criterion='entropy',
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=5,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        random_state=42,
        class_weight=None,
        ccp_alpha=0.001,
        monotonic_cst=None
    )
    trained_decisionTree_model = train_decisionTree_model(decisionTreeModel)
    print(" ")
    
    randomForestModel = model_randomForest_declare(
        n_estimators=200,
        criterion='entropy',
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=16,
        min_weight_fraction_leaf=0.0,
        max_features=0.5,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        random_state=42,
        n_jobs=-1,
        class_weight=None,
        ccp_alpha=0.001,
        monotonic_cst=None
    )
    trained_randomForest_model = train_randomForest_model(randomForestModel)
    print(" ")

    while True:
        choiceProcess = choose_process()
        if choiceProcess == 2:
            choiceAlgorithm = choose_algorithm_PE()
            if choiceAlgorithm == 2:
                choiceModel = choose_model()
                if choiceModel == 0:
                    continue
                elif choiceModel == 2:
                    print("Please choose your criterion:")
                    print("  1. gini")
                    print("  2. entropy")
                    print("  3. log_loss")
                    inputChoice = input("Enter your choice (1/2/3, default = 1): ").strip()
                    if inputChoice == '2':
                        modelCriterion = 'entropy'
                    elif inputChoice == '3':
                        modelCriterion = 'log_loss'
                    else:
                        modelCriterion = 'gini'

                    print("\nPlease choose your class_weight:")
                    print("  1. None")
                    print("  2. balanced")
                    inputChoice = input("Enter your choice (1/2, default = 1): ").strip()
                    if inputChoice == '2':
                        classWeight = 'balanced'
                    else:
                        classWeight = None
                   
                    inputChoice = str(input("\nEnter your max_depth (integer, enter for default = None): ")).strip()
                    try:
                        val = int(inputChoice)
                        if val > 0:
                            maxDepth = int(inputChoice)
                        else:
                            maxDepth = None
                    except ValueError:
                        maxDepth = None

                    inputChoice = str(input("\nEnter your max_leaf_nodes (integer, enter for default = None): ")).strip()
                    try:
                        val = int(inputChoice)
                        if val > 0:
                            maxLeaf = int(inputChoice)
                        else:
                            maxLeaf = None
                    except ValueError:
                        maxLeaf = None

                    inputChoice = str(input("\nEnter your min_samples_split (integer, enter for default = 2): ")).strip()
                    try:
                        val = int(inputChoice)
                        if val > 0:
                            minSamplesSplit = int(inputChoice)
                        else:
                            minSamplesSplit = 2
                    except ValueError:
                        minSamplesSplit = 2

                    inputChoice = str(input("\nEnter your min_samples_leaf (integer, enter for default = 1): ")).strip()
                    try:
                        val = int(inputChoice)
                        if val > 0:
                            minSamplesLeaf = int(inputChoice)
                        else:
                            minSamplesLeaf = 1
                    except ValueError:
                        minSamplesLeaf = 1

                    inputChoice = str(input("\nEnter your ccp_alpha (float, enter for default = 0.0): ")).strip()
                    try:
                        ccpAlpha =float(inputChoice)
                    except ValueError:
                        ccpAlpha = 0.0
                    
                    randomForestEvaluation(
                        modelCriterion=modelCriterion,
                        maxDepth=maxDepth,
                        maxLeaf=maxLeaf,
                        minSamplesSplit=minSamplesSplit,
                        minSamplesLeaf=minSamplesLeaf,
                        classWeight=classWeight,
                        ccpAlpha=ccpAlpha
                    )
                else:
                    print("n_estimators=200")
                    print("criterion='entropy'")
                    print("max_depth=None")
                    print("min_samples_split=10")
                    print("min_samples_leaf=16")
                    print("min_weight_fraction_leaf=0.0")
                    print("max_features=0.5")
                    print("max_leaf_nodes=None")
                    print("min_impurity_decrease=0.0")
                    print("random_state=42")
                    print("n_jobs=-1")
                    print("class_weight=None")
                    print("ccp_alpha=0.001")
                    print("monotonic_cst=None\n")
                    randomForestEvaluationModel(randomForestModel=randomForestModel)
            elif choiceAlgorithm == 3:
                choiceModel = choose_model()
                if choiceModel == 0:
                    continue
                elif choiceModel == 2:
                    print("Please choose your criterion:")
                    print("  1. gini")
                    print("  2. entropy")
                    print("  3. log_loss")
                    inputChoice = input("Enter your choice (1/2/3, default = 1): ").strip()
                    if inputChoice == '2':
                        modelCriterion = 'entropy'
                    elif inputChoice == '3':
                        modelCriterion = 'log_loss'
                    else:
                        modelCriterion = 'gini'

                    print("\nPlease choose your class_weight:")
                    print("  1. None")
                    print("  2. balanced")
                    inputChoice = input("Enter your choice (1/2, default = 1): ").strip()
                    if inputChoice == '2':
                        classWeight = 'balanced'
                    else:
                        classWeight = None
                   
                    inputChoice = str(input("\nEnter your max_depth (integer, enter for default = None): ")).strip()
                    try:
                        val = int(inputChoice)
                        if val > 0:
                            maxDepth = int(inputChoice)
                        else:
                            maxDepth = None
                    except ValueError:
                        maxDepth = None

                    inputChoice = str(input("\nEnter your max_leaf_nodes (integer, enter for default = None): ")).strip()
                    try:
                        val = int(inputChoice)
                        if val > 0:
                            maxLeaf = int(inputChoice)
                        else:
                            maxLeaf = None
                    except ValueError:
                        maxLeaf = None

                    inputChoice = str(input("\nEnter your min_samples_split (integer, enter for default = 2): ")).strip()
                    try:
                        val = int(inputChoice)
                        if val > 0:
                            minSamplesSplit = int(inputChoice)
                        else:
                            minSamplesSplit = 2
                    except ValueError:
                        minSamplesSplit = 2

                    inputChoice = str(input("\nEnter your min_samples_leaf (integer, enter for default = 1): ")).strip()
                    try:
                        val = int(inputChoice)
                        if val > 0:
                            minSamplesLeaf = int(inputChoice)
                        else:
                            minSamplesLeaf = 1
                    except ValueError:
                        minSamplesLeaf = 1

                    inputChoice = str(input("\nEnter your ccp_alpha (float, enter for default = 0.0): ")).strip()
                    try:
                        ccpAlpha =float(inputChoice)
                    except ValueError:
                        ccpAlpha = 0.0

                    compareModelsEvaluation(
                        modelCriterion=modelCriterion,
                        maxDepth=maxDepth,
                        maxLeaf=maxLeaf,
                        minSamplesSplit=minSamplesSplit,
                        minSamplesLeaf=minSamplesLeaf,
                        classWeight=classWeight,
                        ccpAlpha=ccpAlpha
                    )
                else:
                    print("DECISION TREE: ")  
                    print("criterion='entropy'")
                    print("max_depth=10")
                    print("min_samples_split=2")
                    print("min_samples_leaf=5")
                    print("min_weight_fraction_leaf=0.0")
                    print("max_features=None")
                    print("max_leaf_nodes=None")
                    print("min_impurity_decrease=0.0")
                    print("random_state=42")
                    print("class_weight=None")
                    print("ccp_alpha=0.001")
                    print("monotonic_cst=None\n")
                    print("RANDOM FOREST: ")
                    print("n_estimators=200")
                    print("criterion='entropy'")
                    print("max_depth=None")
                    print("min_samples_split=10")
                    print("min_samples_leaf=16")
                    print("min_weight_fraction_leaf=0.0")
                    print("max_features=0.5")
                    print("max_leaf_nodes=None")
                    print("min_impurity_decrease=0.0")
                    print("random_state=42")
                    print("n_jobs=-1")
                    print("class_weight=None")
                    print("ccp_alpha=0.001")
                    print("monotonic_cst=None\n")
                    compareModelsEvaluationModel(decisionTreeModel=decisionTreeModel, randomForestModel=randomForestModel)
            elif choiceAlgorithm == 0:
                continue
            else:
                choiceModel = choose_model()
                if choiceModel == 0:
                    continue
                elif choiceModel == 2:
                    print("Please choose your criterion:")
                    print("  1. gini")
                    print("  2. entropy")
                    print("  3. log_loss")
                    inputChoice = input("Enter your choice (1/2/3, default = 1): ").strip()
                    if inputChoice == '2':
                        modelCriterion = 'entropy'
                    elif inputChoice == '3':
                        modelCriterion = 'log_loss'
                    else:
                        modelCriterion = 'gini'

                    print("\nPlease choose your class_weight:")
                    print("  1. None")
                    print("  2. balanced")
                    inputChoice = input("Enter your choice (1/2, default = 1): ").strip()
                    if inputChoice == '2':
                        classWeight = 'balanced'
                    else:
                        classWeight = None
                   
                    inputChoice = str(input("\nEnter your max_depth (integer, enter for default = None): ")).strip()
                    try:
                        val = int(inputChoice)
                        if val > 0:
                            maxDepth = int(inputChoice)
                        else:
                            maxDepth = None
                    except ValueError:
                        maxDepth = None

                    inputChoice = str(input("\nEnter your max_leaf_nodes (integer, enter for default = None): ")).strip()
                    try:
                        val = int(inputChoice)
                        if val > 0:
                            maxLeaf = int(inputChoice)
                        else:
                            maxLeaf = None
                    except ValueError:
                        maxLeaf = None

                    inputChoice = str(input("\nEnter your min_samples_split (integer, enter for default = 2): ")).strip()
                    try:
                        val = int(inputChoice)
                        if val > 0:
                            minSamplesSplit = int(inputChoice)
                        else:
                            minSamplesSplit = 2
                    except ValueError:
                        minSamplesSplit = 2

                    inputChoice = str(input("\nEnter your min_samples_leaf (integer, enter for default = 1): ")).strip()
                    try:
                        val = int(inputChoice)
                        if val > 0:
                            minSamplesLeaf = int(inputChoice)
                        else:
                            minSamplesLeaf = 1
                    except ValueError:
                        minSamplesLeaf = 1

                    inputChoice = str(input("\nEnter your ccp_alpha (float, enter for default = 0.0): ")).strip()
                    try:
                        ccpAlpha =float(inputChoice)
                    except ValueError:
                        ccpAlpha = 0.0
                    
                    decisionTreeEvaluation(
                        modelCriterion=modelCriterion,
                        maxDepth=maxDepth,
                        maxLeaf=maxLeaf,
                        minSamplesSplit=minSamplesSplit,
                        minSamplesLeaf=minSamplesLeaf,
                        classWeight=classWeight,
                        ccpAlpha=ccpAlpha
                    )
                else:
                    print("criterion='entropy'")
                    print("max_depth=10")
                    print("min_samples_split=2")
                    print("min_samples_leaf=5")
                    print("min_weight_fraction_leaf=0.0")
                    print("max_features=None")
                    print("max_leaf_nodes=None")
                    print("min_impurity_decrease=0.0")
                    print("random_state=42")
                    print("class_weight=None")
                    print("ccp_alpha=0.001")
                    print("monotonic_cst=None")
                    decisionTreeEvaluationModel(decisionTreeModel=decisionTreeModel)
        elif choiceProcess == 0:
            print("=" * 94)
            print('End Program!!! Thank you for using our program!'.center(94))
            print("=" * 94)
            break
        else:
            choiceAlgorithm = choose_algorithm_predict()
            if choiceAlgorithm == 2:
                run_predictor('random_forest_predictor.py')
            elif choiceAlgorithm == 0:
                continue
            else:
                run_predictor('decision_tree_predictor.py')

if __name__ == '__main__':
    main()