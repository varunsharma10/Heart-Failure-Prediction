# Heart Failure Prediction


# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing Datset
dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[: , -1].values
dataset.head()


#Splitting the dataset into the training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Training Naive Bayes model on Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Confusion matrix and Model Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score1 = accuracy_score(y_test, y_pred)
print('Naive Bayes model Accuracy: ','{:.2f}%'.format(100*score1))
sns.heatmap(cm, annot=True)


#Training K-Nearest Neighbors model on Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#Confusion matrix and model Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score2 = accuracy_score(y_test,y_pred)
print('K-Nearest neighbors model Accuracy: ','{:.2f}%'.format(100*score2))
sns.heatmap(cm, annot=True)


# Training Logistic Regression model on Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Confusion matrix and model Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score3 = accuracy_score(y_test, y_pred)
print('Logistic Regression model Accuracy: ','{:.2f}%'.format(100*score3))
sns.heatmap(cm, annot=True)


#Training Support Vector machine model on Training Set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Confusion matrix and model Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score4 = accuracy_score(y_test, y_pred)
print('Logistic Support Vector machine model Accuracy: ','{:.2f}%'.format(100*score4))
sns.heatmap(cm, annot=True)


#Training Kernel SVM model on training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Confusion matrix and model Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score5 = accuracy_score(y_test, y_pred)
print('Kernel SVM model Accuracy: ','{:.2f}%'.format(100*score5))
sns.heatmap(cm, annot=True)


#Training Decision tree model on Training Set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Confusion matrix and model Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score6 = accuracy_score(y_test, y_pred)
print('Decision tree model Accuracy: ','{:.2f}%'.format(100*score6))
sns.heatmap(cm, annot=True)


#Training Random Forest classification model on training Set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Confusion matrix and model Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score7 = accuracy_score(y_test, y_pred)
print('Random forest classification model Accuracy: ','{:.2f}%'.format(100*score7))
sns.heatmap(cm, annot=True)


#Final Results
print('Naive Bayes model Accuracy: ','{:.2f}%'.format(100*score1),'\n')
print('K Nearest neighbors model Accuracy: ','{:.2f}%'.format(100*score2),'\n')
print('Logistic Regression model Accuracy: ','{:.2f}%'.format(100*score3),'\n')
print('Support Vector machine model Accuracy: ','{:.2f}%'.format(100*score4),'\n')
print('Kernel SVM model Accuracy: ','{:.2f}%'.format(100*score5),'\n')
print('Decision tree model Accuracy: ','{:.2f}%'.format(100*score6),'\n')
print('Random forest classification model Accuracy: ','{:.2f}%'.format(100*score7),'\n')