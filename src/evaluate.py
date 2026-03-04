from sklearn.metrics import classification_report, confusion_matrix , f1_score, precision_score, recall_score


def evaluate_model(y_test , y_pred , model_name = "Model"):


    print(f"Evaluation results for {model_name}:")

    print("precision: ", precision_score(y_test, y_pred))
    print("recall: ", recall_score(y_test, y_pred))
    print("f1_score: ", f1_score(y_test, y_pred))

    print("Classification Report:\n", classification_report(y_test, y_pred))

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))