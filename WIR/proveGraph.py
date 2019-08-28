import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import ListedColormap
import warnings
import matplotlib.pyplot as plt

toBeSorted= []
toBeSortedCHI= []

df = pd.pandas.read_csv("trainchisquare.csv", delimiter=";")
df['dummyCat'] = np.random.choice([0, 1], size=(len(df),), p=[0.5, 0.5])#QUESTA RIGA POI SI RIMUOVE, SERVE SOLO A CREARE UNA COLONNA CON UN VALORE RANDOMICO PER DIMOSTRARE CHE TYPE E QUESTA COLONNA SONO INDIPENDENTI



#Feature Selection
testColumns = ['following','followers','actions', 'is_retweet', 'URLCounted', 'HashtagCounted', 'MensionCounted', 'averageHashtag', 'averageURL', 'wordsCounted', 'SpamWordsCounted', 'dummyCat']
#actions;is_retweet;URLCounted;;$;;HashtagCounted;;$;;MensionCounted

data = df    
#X = data.drop(['Type', 'following'], axis='columns')  #independent columns
#y = data.Type    #target column i.e price range

#sta cosa è perchè talvolta mi dava un errore tipo cannot convert float to string
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)
    
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

X=data.iloc[:, [0,1,2,3,5,6,7,8,9,10,11,12]]
y=data.iloc[:,4]

#QUESTA è CHI2
selector = SelectKBest(chi2, k=10)#QUESTO CI DA LE PRIME K FEATURE IN ORDINE DI IMPORTANZA (O ALMENO QUESTA è L'IPOTESI)
selector.fit(X, y)
print (list(zip(selector.get_support(),testColumns)))


#X_new = selector.transform(X)
#print(X_new.shape)

X.columns[selector.get_support(indices=True)]

# 1st way to get the list
vector_names = list(X.columns[selector.get_support(indices=True)])
#print(vector_names)

#QUESTA è INFO-GAIN
selector2 = SelectKBest(mutual_info_classif, k=10)#QUESTO CI DA LE PRIME K FEATURE IN ORDINE DI IMPORTANZA (O ALMENO QUESTA è L'IPOTESI)
selector2.fit(X,y)
#X_new2 = selector2.transform(X)
#print(X_new2.shape)
X.columns[selector2.get_support(indices=True)]
vector_names2=list(X.columns[selector2.get_support(indices=True)])
#print(vector_names2)

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.20)
'''svclassifier = SVC(kernel='rbf', random_state=0, gamma=.01, C=1)
print(train_x.shape)
svclassifier.fit(train_x, train_y)
print("FATTO")
y_pred = svclassifier.predict(test_x)
print("FATTO")

print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))
print(X)
'''
'''

import matplotlib.pyplot as plt

names = X.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'Scores'])
#Sort the dataframe for better visualization
ns_df_sorted = ns_df.sort_values(['Scores', 'Feat_names'], ascending = [False, True])
print(ns_df_sorted)

names2 = X.columns.values[selector2.get_support()]
scores2 = selector2.scores_[selector2.get_support()]
names_scores2 = list(zip(names2, scores2))
ns_df2 = pd.DataFrame(data = names_scores2, columns=['Feat_names', 'Scores'])
#Sort the dataframe for better visualization
ns_df_sorted2 = ns_df2.sort_values(['Scores', 'Feat_names'], ascending = [False, True])
print(ns_df_sorted2)

print("CHI2")
plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.bar(ns_df_sorted.Feat_names, ns_df_sorted.Scores, color='r', align='center')
#plt.bar(vector_names2, selector2.scores_[indices2[range(10)]], color='r', align='center')
plt.show()

print("info_gain")
plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.bar(ns_df_sorted2.Feat_names, ns_df_sorted2.Scores, color='r', align='center')
#plt.bar(vector_names2, selector2.scores_[indices2[range(10)]], color='r', align='center')
plt.show()
'''
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
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
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
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X.iloc[0:2, 0], X.iloc[0:2, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
