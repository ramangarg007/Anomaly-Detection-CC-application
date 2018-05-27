import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print('Before Scaling')
print(X[:2])

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

print('After Scaling')
print(X[:2])
