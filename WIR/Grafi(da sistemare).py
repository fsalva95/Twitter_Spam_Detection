import os
import shutil
import sys
from sklearn.model_selection import train_test_split
import shutil
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings


filename = "data_train.txt"


X = []  #intero dataset
Y = []  #groundtruth

X1 = []  #intero dataset
Y1 = []  #groundtruth



train_x = []  #attributi learning
train_y = []  #ground truth
test_x = []
test_y = []

train1_x = []  #attributi learning
train1_y = []  #ground truth
test1_x = []
test1_y = []


with open(filename, 'r',encoding="utf8") as file:

        i=0
        for row in file:
                if i!=0 and len(row.split(";;$;;"))==12:
                        value=row.split(";;$;;")
                        classification=value[4]
                        value=list(map(str.strip,value))
                        Y1.append(classification)

                        value.remove(value[4])
                        value=list(map(float,value))
                        value=np.asarray(value)
                        value=value.astype(int)

                        value1=value
                        
                        value=np.delete(value,10) #follower e followees
                        value=np.delete(value,9)
                        value=np.delete(value,8)
                        value=np.delete(value,7)
                        value=np.delete(value,6)
                        value=np.delete(value,5)
                        value=np.delete(value,4)
                        value=np.delete(value,3)
                        value=np.delete(value,2)

                
                        value1=np.delete(value1,0)#mension e hashtag counted
                        value1=np.delete(value1,0)
                        value1=np.delete(value1,0)
                        value1=np.delete(value1,0)
                        value1=np.delete(value1,0)
                        value1=np.delete(value1,2)
                        value1=np.delete(value1,2)
                        value1=np.delete(value1,2)
                        value1=np.delete(value1,2)


                        X1.append(value1)
                        
                        X.append(value)
                        Y.append(classification)
                                
                       
                i=i+1


X=np.asarray(X)
X1=np.asarray(X1)
y=[] 
y1=[]

for i in range(0,len(X)) :
        if Y[i]=='Quality': #ROSSI SPAM #BLU QUALITY
                y.append(0)
        else:
                y.append(1)

for j in range(0,len(X1)) :
    if Y1[j]=='Quality':
        y1.append(0)
    else :
        y1.append(1)


scaler= MinMaxScaler((0,1))
X= scaler.fit_transform(X)
X1= scaler.fit_transform(X1)



train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.20)
train1_x, test1_x, train1_y, test1_y = train_test_split(X1, y1, test_size = 0.20)


def make_meshgrid(x, y, h=.02):

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max,h),
                         np.arange(y_min, y_max,h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)

    return out



models = (SVC(kernel='rbf',gamma=1000,C=1),
          SVC(kernel='rbf',gamma=1000,C=1)) 


models[0].fit(X,y)

models[1].fit(X1,y1)


y_pred=models[0].predict(test_x)
print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))

y1_pred=models[1].predict(test1_x)
print(confusion_matrix(test1_y,y1_pred))
print(classification_report(test1_y,y1_pred))



# title for the plots
titles = ('SVM with RBF kernel(follower and following)',
          'SVM with RBF kernel1(mension and hashtag counted)')

# Set-up 1x2 grid for plotting.
fig, sub = plt.subplots(1, 2)
#plt.subplots_adjust(wspace=0.4, hspace=0.4)


levels = np.linspace(0, 1, 100)

X_0, X_1 = test_x[:, 0], test_x[:, 1]
xx, yy = make_meshgrid(X_0, X_1)


X1_0, X1_1 = test1_x[:, 0], test1_x[:, 1]
xx1, yy1 = make_meshgrid(X1_0, X1_1)


for title, ax in zip(titles, sub.flatten()):
        if title=='SVM with RBF kernel(follower and following)':
                plot_contours(ax, models[0], xx, yy,cmap=plt.cm.coolwarm, alpha=1,levels=levels)
                ax.scatter(X_0, X_1, c=test_y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xlabel('Following')
                ax.set_ylabel('Followers')
        else:
                plot_contours(ax, models[1], xx1, yy1,cmap=plt.cm.coolwarm, alpha=1,levels=levels)
                ax.scatter(X1_0, X1_1, c=test1_y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
                ax.set_xlim(xx1.min(), xx1.max())
                ax.set_ylim(yy1.min(), yy1.max())
                ax.set_xlabel('Hashtag counted')
                ax.set_ylabel('Mension counted')

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
            

plt.show()












