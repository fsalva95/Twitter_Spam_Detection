import os
import shutil
import sys  ########
from sklearn.model_selection import train_test_split
import shutil
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
#from mlxtend.plotting import plot_decision_regions


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



# reading csv file
with open(filename, 'r',encoding="utf8") as file:

        i=0
        for row in file:
                if i!=0 and len(row.split(";;$;;"))==12 and i<3000:# i<500 per vedere che il codice funziona (è da togliere per trainare tutto il db
                        value=row.split(";;$;;")
                        classification=value[4]
                        value=list(map(str.strip,value))
                        Y1.append(classification)
                        #Y.append(classification)

                        value.remove(value[4])  #rimuovo classificazione
                        value=list(map(float,value))
                        value=np.asarray(value)
                        value=value.astype(int)
                        #print(value.shape)

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

                        if value[1]<2000 and value[0]<2000: #follower e followees (da togliere se si sta provando altro)
                        
                                X.append(value)
                                Y.append(classification)
                                
                       
                i=i+1


print("FATTO")
X=np.asarray(X)
X1=np.asarray(X1)
y=[]
y1=[]
for i in range(0,len(X)) :
    if Y[i]=='Quality':
        y.append(0.0)
    else :
        y.append(1.0)

for j in range(0,len(X1)) :
    if Y1[j]=='Quality':
        y1.append(0.0)
    else :
        y1.append(1.0)

print(len(Y1))
print(len(X1))
print(len(Y))
print(len(X))
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.20)
train1_x, test1_x, train1_y, test1_y = train_test_split(X1, y1, test_size = 0.20)


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max),
                         np.arange(y_min, y_max))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out




# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1  # SVM regularization parameter
models = (svm.SVC(kernel='rbf', gamma="auto", C=C),
          svm.SVC(kernel='rbf', gamma="auto", C=C)) #gamma era 0.7



model1=models[0].fit(X,y)
model2=models[1].fit(X1,y1)


y_pred=models[0].predict(test_x)
print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))

y1_pred=models[1].predict(test1_x)
print(confusion_matrix(test1_y,y1_pred))
print(classification_report(test1_y,y1_pred))



# title for the plots
titles = ('SVC with RBF kernel(follower and following)',
          'SVC with RBF kernel1(mension and hashtag counted')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)


print("##############################FATTO#############################")


X_0, X_1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X_0, X_1)


X1_0, X1_1 = X1[:, 0], X1[:, 1]
xx1, yy1 = make_meshgrid(X1_0, X1_1)


for title, ax in zip(titles, sub.flatten()):
        if title=='SVC with RBF kernel(follower and following)':
                plot_contours(ax, model1, xx, yy,cmap=plt.cm.coolwarm, alpha=1)#alpha cambia intensità colore (era 0.8)
                ax.scatter(X_0, X_1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xlabel('Following')
                ax.set_ylabel('Followers')
        else:
                plot_contours(ax, model2, xx1, yy1,cmap=plt.cm.coolwarm, alpha=1)
                ax.scatter(X1_0, X1_1, c=y1, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
                ax.set_xlim(xx1.min(), xx1.max())
                ax.set_ylim(yy1.min(), yy1.max())
                ax.set_xlabel('Hashtag counted')
                ax.set_ylabel('Mension counted')

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        print("##############################FATTOOOO#############################")
            

plt.show()












