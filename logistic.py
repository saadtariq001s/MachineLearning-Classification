from distutils.errors import PreprocessError
from re import L
import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt

df = pd.read_csv("ChurnData.csv")

churn_df = df[["tenure","age", "address", "income", "ed", "employ","churn"]]
churn_df["churn"] = churn_df["churn"].astype(int)

x = np.asanyarray(churn_df[["tenure","age", "address", "income", "ed", "employ"]])
y = np.asanyarray(churn_df["churn"])

# normalising dataset

from sklearn import preprocessing

x = preprocessing.StandardScaler().fit(x).transform(x)

# modelling

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.2,random_state=4)
print("Training data size is:", train_x.shape, train_y.shape)
print("Testing data size is:", test_x.shape, test_y.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

lr = LogisticRegression(C=0.01, solver='liblinear').fit(train_x, train_y)

lr_hat = lr.predict(test_x)

print(lr_hat)

# calculating probability

yprob = lr.predict_proba(test_x)

print(yprob\n) 

# ^ first coloum mein class 0 ki probability, next mein, class 1 ki.

from sklearn.metrics import jaccard_score, classification_report

print(jaccard_score(test_y,lr_hat, pos_label=0))
print(classification_report(test_y,lr_hat))


