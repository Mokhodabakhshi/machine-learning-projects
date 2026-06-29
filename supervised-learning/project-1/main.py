import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_excel("CSM_dataset.xlsx",na_values="NaN").set_axis(list(range(14)),axis=1)

dist_attr = [0]
cont_attr = [x for x in range(1,14)]

a = list(data.isna().sum())
nan_value = [i for i,_ in enumerate(a) if _ != 0]

# missing values
for i in nan_value:
    _mean = data[i].mean()
    data[i] = data[i].replace(math.nan, _mean)

attr = [x for x in cont_attr if x != 4]

X = data[attr]
Y = data.pop(4)

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)


std = StandardScaler()

x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

lr_sgd = SGDRegressor()
lr_sgd.fit(x_train, y_train)
score = lr_sgd.score(x_test,y_test)

y_pred = lr_sgd.predict(x_test)

mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
print(f"score is : {score}")
print(f"mse is : {mse}")