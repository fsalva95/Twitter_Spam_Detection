import os
import shutil
import sys  ########
from sklearn.model_selection import train_test_split
import shutil
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
#from mlxtend.plotting import plot_decision_regions


filename = "data_train.txt"


X = []  #intero dataset
Y = []  #groundtruth

train_x = []  #attributi learning
train_y = []  #ground truth
test_x = []
test_y = []



# reading csv file
with open(filename, 'r',encoding="utf8") as file:

        i=0
        for row in file:
                if i!=0 and len(row.split(";;$;;"))==12:
                        value=row.split(";;$;;")
                        
                        classification=value[4]
                        value=list(map(str.strip,value))
                        Y.append(classification)

                        value.remove(value[4])  #rimuovo classificazione
                        value=list(map(float,value))
                        value=np.asarray(value)

                        
                        X.append(value)

                i=i+1


X=np.asarray(X)
y=[]
for i in range(0,len(X)) :
    if Y[i]=='Quality':
        y.append(0.0)
    else :
        y.append(1.0)


scaler= MinMaxScaler((0,1))
X= scaler.fit_transform(X)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.20)
svclassifier = SVC(kernel='rbf',gamma=1000,C=1)
print(train_x.shape)
svclassifier.fit(train_x, train_y)

y_pred = svclassifier.predict(test_x)


print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))














