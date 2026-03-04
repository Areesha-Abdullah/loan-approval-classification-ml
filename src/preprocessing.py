#1--first we load data 
#2--separate target variable from features
#3--we deal with missing values
#4--we encode categorical variables

import pandas as pd
def preprocess_data(data_path):
    # 1--
    df = pd.read_csv(data_path)

    # Drop rows with missing target
    df = df.dropna(subset=[" loan_status"])
    # 2--
    y = df[' loan_status'].map({' Approved': 1, ' Rejected': 0})
    X = df.drop(" loan_status" , axis=1)

    # 3--

    for col in X.columns:
        if X[col].dtype == object:
            X[col].fillna(X[col].mode()[0])

        else: 
            X[col].fillna(X[col].median())


    # 4--
    X = pd.get_dummies(X , drop_first=True)

    return X, y