import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

df = pd.read_csv("cell_samples.csv")
print(df.columns)

cell = df[df["Class"]==4][0:50].plot(kind='scatter',x='Clump',y='UnifSize',color="DarkBlue", label="Malignant");
df[df['Class']==2][0:50].plot(kind="scatter", x="Clump",y="UnifSize", color="Yellow", label= "Benign", ax=cell);


# necesarry to have all coloumns with integar datatypes
"""
checking if all are integars or not, if not then 
converting them to integar
"""

# print(df.dtypes)

''' BareNuc is not integar, so lets make it
'''

df = df[pd.to_numeric(df["BareNuc"], errors="coerce").notnull()]
df["BareNuc"] = df["BareNuc"].astype(int)
#print(df.dtypes)

x = np.asanyarray(df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize','BareNuc', 'BlandChrom', 'NormNucl', 'Mit']])
print(x[0:5])
y = np.asanyarray(df["Class"].astype(int))
print(y[0:5])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=4)

print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)

from sklearn import svm
clf = svm.SVC(kernel="rbf")
clf.fit(x_train, y_train)

y_hat = clf.predict(x_test)

from sklearn.metrics import f1_score,jaccard_score,classification_report

print("F1 score is:", f1_score(y_test, y_hat, average="weighted"))
print("Accuracy is:", jaccard_score(y_test, y_hat, pos_label=2))
print(classification_report(y_test, y_hat))

plt.show()