#importing libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

#splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initializing and training the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

#evaluating the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

accuracy
