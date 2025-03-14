from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

print(f'The data has a shape of {X.shape}, and the target has a shape of {y.shape}')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'The training set has {X_train.shape[0]} datapoints and the test set has {X_test.shape[0]} datapoints.')

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(f'The max of training data is {X_train.max():.2f} and the min is {X_train.min():.2f}.')

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear')
clf = model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f'The accuracy is {accuracy:.4f}')


import numpy as np

raw = clf.coef_ @ X_test.T + clf.intercept_
prob = 1 / (1 + np.exp(-raw))
y_pred = np.where(prob > 0.5, 1, 0)
my_accuracy = 0
for index, y in enumerate(y_pred.T):
    if y_test[index] == y:
      my_accuracy += 1
my_accuracy /= len(y_test)


print(f'My accuracy is {my_accuracy:.4f} and the accuracy calculated from built-in function is {accuracy:.4f}')