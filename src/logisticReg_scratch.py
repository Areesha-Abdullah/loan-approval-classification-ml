import numpy as np
class LogisticRegressionFromScratch:

    def __init__(self , learning_rate=0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None


    def sigmoid(self , z):
        # numerical stability
        z = np.clip(z, -500, 500)
        return 1 / (1+np.exp(-z))

    def fit(self , X_train ,y_train):
        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=float)
        # y = y.reshape(-1)
    
        # adding a column of ones  to X for bias term so that we can do dot product of W.X
        X = np.insert(X_train ,0 , 1 , axis=1)
        
        #initializing weights and bias

        self.weights = np.ones(X.shape[1])

        for  _ in range(self.epochs):
            y_hat = self.sigmoid(np.dot(X , self.weights))
            gradient = np.dot((y_train-y_hat),X)/X.shape[0]
            self.weights = self.weights +self.learning_rate  * gradient

        
    def predict_probability(self, X_test):
        X_test = np.array(X_test, dtype=float)
        X = np.insert(X_test ,0 , 1 , axis=1)
        y_hat = self.sigmoid(np.dot(X , self.weights))
        return y_hat
    
    def predict(self , X_test, threshold = 0.5):
        probs = self.predict_probability(X_test)
        return (probs >= threshold).astype(int)



