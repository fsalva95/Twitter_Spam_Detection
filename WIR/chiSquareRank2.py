import pandas as pd
import numpy as np

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


