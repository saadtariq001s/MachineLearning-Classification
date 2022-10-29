from random import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


path = 'abc.csv'
df = pd.read_csv(path)

#print(df.head())

#print(df['custcat'].value_counts())

# df.hist(column='income', bins=50)
# plt.show()

#print(df.columns)

# Index(['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
#       'employ', 'retire', 'gender', 'reside', 'custcat'],
#      dtype='object')

x = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']]
y = df['custcat'].values

x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

# Data Standardization gives the data zero mean and unit variance, 
# it is good practice, especially for algorithms such as KNN which
# is based on the distance of data points

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=4)

print("Train Set:", x_train.shape, y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

# k = 6

# Train model predict

# neigh = KNeighborsClassifier(n_neighbors= 6).fit(x_train, y_train)

# model

# y_hat = neigh.predict(x_test)

# evaluation

k = 10
mean_acc = np.zeros((k-1))
std_acc = np.zeros((k-1))

from sklearn import metrics 

for n in range(1,k):
    neigh = KNeighborsClassifier(n_neighbors= n).fit(x_train, y_train)
    y_hat = neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, y_hat)

print(mean_acc)

#plotting

plt.plot(range(1,k), mean_acc, 'g')
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")

plt.show()



# from sklearn import metrics

# print("Training Set Accuracy:", metrics.accuracy_score(y_train, neigh.predict(x_train)))
# print("Testing Set Accuracy:", metrics.accuracy_score(y_test, y_hat))
