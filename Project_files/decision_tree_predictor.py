import os
import joblib
import pandas as pd

def load_resources():
    model_file = 'decision_tree_stroke_model.pkl'
    prep_file = 'preprocessor.pkl'

    if not os.path.exists(model_file):
        print(
            f"Error: '{model_file}' is not found. Please run decision_tree_train.py!"
        )
        exit(1)
    if not os.path.exists(prep_file):
        print(
            f"Error: '{prep_file}' is not found. Please run preprocess.py!"
        )
        exit(1)

    model = joblib.load(model_file)
    prep_data = joblib.load(prep_file)

    return model, prep_data['preprocessor'], prep_data['feature_names']

def get_natural_user_input():
    print("\n" + "=" * 94)
    print(' ENTER PATIENT INFORMATION (ACTUAL/NATURAL VALUES) '.center(94))
    print("=" * 94)

    # Enter real values
    age = float(input("\n1. Age (e.g., 67): "))
    avg_glucose_level = float(input("\n2. Average glucose level in blood (e.g., 228.69): "))
    bmi = float(input("\n3. BMI (e.g., 36.6): "))

    # Enter binary medical history data (0 or 1)
    print("\n--- Medical history ---")
    hyp_choice = input("\nDoes the patient have high blood pressure? (1. Yes / 2. No): ").strip()
    hypertension = 1 if hyp_choice == '1' else 0

    heart_choice = input("\nDoes the patient have a heart disease? (1. Yes / 2. No): ").strip()
    heart_disease = 1 if heart_choice == '1' else 0

    # Enter natural category values
    print("\n--- Category information ---")

    g_choice = input("\nGender (1. Female / 2. Male): ").strip()
    gender = 'Male' if g_choice == '2' else 'Female'

    m_choice = input("\nHas the patient ever been married? (1. Yes / 2. No): ").strip()
    ever_married = 'Yes' if m_choice == '1' else 'No'

    print("\nWork type:")
    print(
        " 1. Private \n 2. Self-employed \n 3. Govt_job \n 4. children \n 5. Never_worked "
    )
    w_dict = {
        '1': 'Private',
        '2': 'Self-employed',
        '3': 'Govt_job',
        '4': 'children',
        '5': 'Never_worked',
    }
    work_type = w_dict.get(input("Choose (1-5): ").strip(), 'Private')

    r_choice = input("\nResidence type (1. Urban / 2. Rural): ").strip()
    Residence_type = 'Urban' if r_choice == '1' else 'Rural'

    print("\nSmoking status:")
    print(
        " 1. Never smoked \n 2. Formerly smoked \n 3. Smokes \n 4. Unknown "
    )
    s_dict = {
        '1': 'never smoked',
        '2': 'formerly smoked',
        '3': 'smokes',
        '4': 'Unknown',
    }
    smoking_status = s_dict.get(input("Choose (1-4): ").strip(), 'never smoked')

    # Package the raw data as an initial DataFrame (containing all input columns for the preprocessor)
    raw_df = pd.DataFrame([
        {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status,
        }
    ])

    return raw_df

def predict_stroke():
    model, preprocessor, feature_names = load_resources()

    while True:
        # 1. Collect data from users
        raw_df = get_natural_user_input()

        # 2. Use the exact preprocessor that was fitted in the preprocess.py to perform the transformation.
        processed_array = preprocessor.transform(raw_df)

        # 3. Create a DataFrame that matches the list of feature columns exactly
        input_df = pd.DataFrame(processed_array, columns=feature_names)

        # 4. Prediction
        prediction = model.predict(input_df)[0]

        print("\n" + "*" * 94)
        if prediction == 1:
            print(' PREDICTION RESULT: HIGH risk of stroke (Stroke = 1) '.center(94))
        else:
            print(' PREDICTION RESULT: LOW risk / NO stroke (Stroke = 0) '.center(94))
        print("*" * 94)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            print(f"Probability of not having a stroke (0): {probs[0]*100:.2f}%")
            print(f"Probability of having a stroke (1): {probs[1]*100:.2f}%")

        print("*" * 94)
        cont = (
            input("Continue with a prediction for another patient? (y/n): ")
            .strip()
            .lower()
        )
        if cont != 'y':
            break

if __name__ == '__main__':
    predict_stroke()