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

train_x = []  #attributi learning
train_y = []  #ground truth
test_x = []
test_y = []



# reading csv file
with open(filename, 'r',encoding="utf8") as file:

        i=0
        for row in file:
                if i!=0 and len(row.split(";;$;;"))==12 and i<3000:# i<500 per vedere che il codice funziona (è da togliere per trainare tutto il db
                        value=row.split(";;$;;")
                        classification=value[4]
                        value=list(map(str.strip,value))
                        Y.append(classification)

                        value.remove(value[4])  #rimuovo classificazione
                        value=list(map(float,value))
                        value=np.asarray(value)
                        value=value.astype(int)
                        #print(value.shape)
                        '''
                        value=np.delete(value,10) #follower e followees
                        value=np.delete(value,9)
                        value=np.delete(value,8)
                        value=np.delete(value,7)
                        value=np.delete(value,6)
                        value=np.delete(value,5)
                        value=np.delete(value,4)
                        value=np.delete(value,3)
                        value=np.delete(value,2)
                        '''
                        '''
                        value=np.delete(value,0)#action e url count
                        value=np.delete(value,0)
                        value=np.delete(value,1)
                        value=np.delete(value,1)
                        value=np.delete(value,2)
                        value=np.delete(value,2)
                        value=np.delete(value,2)
                        value=np.delete(value,2)
                        value=np.delete(value,2)
                        '''
                        
                        value=np.delete(value,0)#mension e hashtag counted
                        value=np.delete(value,0)
                        value=np.delete(value,0)
                        value=np.delete(value,0)
                        value=np.delete(value,0)
                        value=np.delete(value,2)
                        value=np.delete(value,2)
                        value=np.delete(value,2)
                        value=np.delete(value,2)
                        

                        #if value[1]<10000 and value[0]<10000: #follower e followees (da togliere se si sta provando altro)
                        
                        X.append(value)
                       
                        #print(row)
                i=i+1


#print(X) #troppo pesante
print("FATTO")
#print(Y)
X=np.asarray(X)
y=[]
for i in range(0,len(X)) :
    if Y[i]=='Quality':
        y.append(0.0)
    else :
        y.append(1.0)


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.20)
svclassifier = SVC(kernel='rbf',gamma="auto")
print(train_x.shape)
svclassifier.fit(train_x, train_y)
print("FATTO")
#y_pred = svclassifier.predict(test_x)
#print("FATTO")

#print(confusion_matrix(test_y,y_pred))
#print(classification_report(test_y,y_pred))

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
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=1, C=C)) #gamma era 0.7
m=models[2]
models = (clf.fit(X, y) for clf in models)
m.fit(X,y)

y_pred = m.predict(test_x)
print("FATTO")

print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)


print("##############################FATTO#############################")


df0=[]; #QUALITY
df1=[]; #SPAM
for i in range(0,len(X)) :
    if Y[i]=='Quality':
        df0.append(X[i])
    else :
        df1.append(X[i])
df0=np.asarray(df0)
df1=np.asarray(df1)



X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
        if title=='SVC with RBF kernel':
                plot_contours(ax, m, xx, yy,cmap=plt.cm.coolwarm, alpha=1)#alpha cambia intensità colore
        else:
                plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        print("##############################FATTOOOO#############################")
            

plt.show()












