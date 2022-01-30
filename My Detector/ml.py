from keras.models import Sequential
from keras.layers import Dense
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


class DNN:
    def __init__(self, X_train, y_train):
        self.model = Sequential()
        # Add input layer
        self.model.add(Dense(64, activation='relu'))
        # Add output layer
        self.model.add(Dense(1))

        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

        self.model.fit(X_train, y_train)

    def predict(self, x):
        return self.model.predict(x)


class SVM:
    def __init__(self, X_train, y_train):
        self.clf = svm.SVC(kernel='linear')  # Linear Kernel
        self.clf.fit(X_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class LR:
    def __init__(self, X_train, y_train):
        self.clf = LinearRegression()
        self.clf.fit(X_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    # def get_params(self, deep):
    #     return self.reg.get_params(deep)
    #
    # def set_params(self, ):


class DT:
    def __init__(self, X_train, y_train):
        self.clf = DecisionTreeClassifier(random_state=0)
        self.clf.fit(X_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

# # test training example
# svm1 = SVM([[1, 2, 3, 4], [2, 2, 2, 4], [9, 2, 3, 7], [4, 4, 4, 4]], [10, 10, 21, 16])
# print(svm1.predict([[2, 2, 2, 1]])) # result = 10
# print(svm1.predict([[200, 2, 2, 1]])) # result = 21

