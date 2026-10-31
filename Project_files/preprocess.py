import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

def system_preprocessing(path):
    # Read data & Clean, remove IDs, and eliminate anomalous gender entries
    df = pd.read_csv(path)
    df = df.drop(columns=['id'])
    df = df[df['gender'] != 'Other']

    # Separarte X and y 
    X = df.drop(columns=['stroke'])
    y = df['stroke']

    # Column classification
    num_cols_skewed = ['avg_glucose_level', 'bmi']  
    num_cols_normal = ['age']  # 
    binary_cols = ['hypertension', 'heart_disease'] 
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']  

    # Pipeline for preprocessing
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

    # Fit and transform all data in X
    X_preprocessed = preprocessor.fit_transform(X)

    # Get the complete list of column names
    encoded_cat_cols = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols))
    feature_names = num_cols_skewed + num_cols_normal + binary_cols + encoded_cat_cols

    # Handling imbalance of data by using SMOTE on the ENTIRE DATASET
    categorical_feature_names = binary_cols + encoded_cat_cols
    cat_indices = [feature_names.index(col) for col in categorical_feature_names]
    smotenc = SMOTENC(categorical_features=cat_indices, random_state=42)
    X_resampled, y_resampled = smotenc.fit_resample(X_preprocessed, y)

    # Export result file
    resampled_df = pd.DataFrame(X_resampled, columns=feature_names)
    resampled_df['stroke'] = y_resampled.values if hasattr(y_resampled, 'values') else y_resampled
    resampled_df.to_csv('stroke_preprocessed.csv', index=False)
    print("-> Saved preprocessed data in stroke_preprocessed.csv")

    # The .pkl file is used to standardize data entered by users
    # This section is intended solely for the future deployment of the PREDICTOR
    joblib.dump({
        'preprocessor': preprocessor,
        'feature_names': feature_names
    }, 'preprocessor.pkl')

if __name__ == '__main__':
    system_preprocessing('healthcare-dataset-stroke-data.csv')