import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import OrdinalEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

raw_data = pd.read_csv("BOM.csv").set_axis(list(range(23)),axis=1)
raw_data = raw_data.drop(0,axis=1)

dist_attr = [1,7,9,10,21,22]
cont_attr = [x for x in range(1,23) if x not in dist_attr]

for index in range(1,23):
    if index in cont_attr:
        _mean = raw_data[index].mean()
        raw_data[index] = raw_data[index].replace(math.nan, _mean)
    else:
        _mode = raw_data[index].mode()[0]
        raw_data[index] = raw_data[index].replace(math.nan, _mode)
    
    
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(raw_data[cont_attr].values)
normal = pd.DataFrame(x_scaled, columns=cont_attr, index=raw_data.index)
raw_data[cont_attr] = normal

ord_enc = OrdinalEncoder()
binary_data = [21,22]
raw_data[binary_data] = ord_enc.fit_transform(raw_data[binary_data])
Y = raw_data.pop(22)

cat_data = [x for x in dist_attr if x not in binary_data]
one_hot_enc = pd.get_dummies(raw_data[cat_data])
raw_data = raw_data.drop(cat_data,axis=1)
X = raw_data.join(one_hot_enc)

X.columns = X.columns.astype(str)

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

best_k = 0
best_f1_score = 0

for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    
    y_pred = knn.predict(x_test)
    f1 = f1_score(y_test,y_pred)
    
    if f1 > best_f1_score:
        best_f1_score = f1
        best_k = k
print(f"Best K is: {best_k}, best f1 : {best_f1_score}")
