import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")

raw_data = pd.read_csv("anti-malware.csv")
attr = list(raw_data.columns.values)

Y = np.array(raw_data.pop('OUTPUT'))
X = np.array(raw_data)

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.transform(x_test)


#----------- Logistic Regression ------------------
lg = LogisticRegression()
lg.fit(x_train, y_train)
y_pred = lg.predict(x_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Logistic Accuracy:", acc)
plt.title(f"Logistic Regression Confusion Matrix, Accuracy : {acc}")
sns.heatmap(cm,annot=True,cmap="Blues")
plt.show()


#----------- Gaussian Naive Bayes ------------------
gnb = GaussianNB()

gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("Gaussian Naive Bayes Accuracy:", acc)

cm = confusion_matrix(y_test, y_pred)
plt.title(f"Gaussian Naive Bayes Confusion Matrix, Accuracy : {acc}")
sns.heatmap(cm, annot=True, cmap="Blues")
plt.show()