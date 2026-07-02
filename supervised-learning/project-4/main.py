from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

data = fetch_olivetti_faces()

X = data.data
Y = data.target

print(f"Number of people {len(set(Y[:]))}")

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

#------------------ One of each -----------------------

# i = np.random.randint(0,10)
# while True:
#     if i <= 399:
#         plt.imshow(X[i].reshape(64,64),cmap="gray")
#         plt.title(f"Person {Y[i]}")
#         plt.show()
#         i += 10
#     else:
#         break
    
#----------------------------------------------
fig, axes = plt.subplots(5, 8, figsize=(12, 8))
i = 0
while True:
    if i <= 399:
        ax = axes.ravel()[i // 10]
        ax.imshow(X[i].reshape(64,64), cmap="gray")
        ax.set_title(f"{Y[i]}")
        ax.axis("off")
        i += 10
    else:
        break
plt.tight_layout()
plt.show()

#------------- Support Vector Machine-------------
svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Support Vector Machine Accuracy : {acc}")

#-------------- Ada Boost Classifier -------------
ada_boost = AdaBoostClassifier()
ada_boost.fit(x_train, y_train)
y_pred = ada_boost.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Ada Boost Accuracy : {acc}")

#-------------- Random Forest Classifier ------------

random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy : {acc}")