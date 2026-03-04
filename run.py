from src.preprocessing import preprocess_data
from src.evaluate import evaluate_model
from src.train import train_logistic_model, train_logisticfromScratch_model, train_decision_tree
from src.SMOTE import apply_smote

Data_path = "data/loan_approval_dataset.csv"

def main():
    X , y = preprocess_data(Data_path)


# ------------logistic regression from scratch-----------------
    y_test , y_pred = train_logisticfromScratch_model(X,y)
    # print(y_pred)
    evaluate_model(y_test , y_pred, model_name="Logistic Regression from Scratch")


#-------------Without SMOTE-----------------
# -------------logistic regression-----------------
    y_test , y_pred = train_logistic_model(X,y)

    # print(y_pred)
    evaluate_model(y_test , y_pred, model_name="Logistic Regression")

#-------------decision tree-----------------

    y_test , y_pred = train_decision_tree(X,y)
    # print(y_pred)
    evaluate_model(y_test , y_pred, model_name="Decision Tree")

#-------------with SMOTE-----------------

    X_smote , y_smote = apply_smote(X,y)
    y_test_lr_s , y_pred_lr_s = train_logistic_model(X_smote,y_smote)
    evaluate_model(y_test_lr_s , y_pred_lr_s, model_name="Logistic Regression with SMOTE")

    y_test_dt_s , y_pred_dt_s = train_decision_tree(X_smote,y_smote)
    evaluate_model(y_test_dt_s , y_pred_dt_s, model_name="Decision Tree with SMOTE")

if __name__ == "__main__":
    main()
