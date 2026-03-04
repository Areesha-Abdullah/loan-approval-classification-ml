from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.logisticReg_scratch import LogisticRegressionFromScratch

from sklearn.tree import DecisionTreeClassifier


def train_logisticfromScratch_model(X,y):

    X_train , X_test, y_train , y_test = train_test_split(X,y, test_size= 0.2, random_state=42 , stratify= y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegressionFromScratch(learning_rate=0.01, epochs=1000)
    model.fit(X_train , y_train)
    y_pred = model.predict(X_test)
    return y_test , y_pred


def train_logistic_model(X,y):

    X_train , X_test, y_train , y_test = train_test_split(X,y, test_size= 0.2, random_state=42 , stratify= y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Stratify ensures that the train and test sets keep the same proportion of each class as the original dataset. This is especially important for imbalanced classification problems, because it prevents one class from being over-represented or missing in the test set, making model evaluation more reliable.
    model = LogisticRegression(max_iter  = 1000) 

    model.fit(X_train , y_train)

    y_pred = model.predict(X_test)

    return y_test , y_pred

def train_decision_tree(X,y):
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2, random_state = 42, stratify=y)

    model = DecisionTreeClassifier(
        random_state = 42,
        max_depth = 5
    )

    model.fit(X_train , y_train)
    y_pred = model.predict(X_test)

    return y_test , y_pred