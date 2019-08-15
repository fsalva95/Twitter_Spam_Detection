import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency
from decimal import Decimal

toBeSorted= []
toBeSortedCHI= []

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        #tmp=Decimal(self.p)
        #tmp2=round(tmp,25)
        #print("p-value of {0}".format(colX))
        #print(self.p)
        #print("chi2 of {0}".format(colX))
        #print(self.chi2)
        #print(result)
        #print("degree of freedom:")
        #print(self.dof)
        toBeSorted.append((colX, self.p))
        toBeSortedCHI.append((colX, self.chi2))
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)

df = pd.pandas.read_csv("trainchisquare.csv", delimiter=";")
df['dummyCat'] = np.random.choice([0, 1], size=(len(df),), p=[0.5, 0.5])#QUESTA RIGA POI SI RIMUOVE, SERVE SOLO A CREARE UNA COLONNA CON UN VALORE RANDOMICO PER DIMOSTRARE CHE TYPE E QUESTA COLONNA SONO INDIPENDENTI

#Initialize ChiSquare Class
cT = ChiSquare(df)

#Feature Selection
testColumns = ['following','followers','actions', 'is_retweet', 'URLCounted', 'HashtagCounted', 'MensionCounted', 'averageHashtag', 'averageURL', 'wordsCounted', 'SpamWordsCounted', 'dummyCat']
#actions;is_retweet;URLCounted;;$;;HashtagCounted;;$;;MensionCounted
test20=['actions']
for var in testColumns:
    cT.TestIndependence(colX=var,colY="Type" )  
toBeSorted.sort(key = lambda x: x[1])
print("\n")
print(toBeSorted)
print("\n")
#print("NOW THE CHI2")
#print(toBeSortedCHI)
toBeSortedCHI.sort(key = lambda x: x[1])
toBeSortedCHI.reverse()
print("\n")
print("THE RANK OF ATTRIBUTES IS (chi2):")
print(toBeSortedCHI)
print("\n")


data = df    
#X = data.drop(['Type', 'following'], axis='columns')  #independent columns
#y = data.Type    #target column i.e price range
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)
    
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

X=data.iloc[:, [0,1,2,3,5,6,7,8,9,10,11,12]]
y=data.iloc[:,4]

#QUESTA è CHI2
selector = SelectKBest(chi2, k=3)#QUESTO CI DA LE PRIME K FEATURE IN ORDINE DI IMPORTANZA (O ALMENO QUESTA è L'IPOTESI)
selector.fit(X, y)

X_new = selector.transform(X)
print(X_new.shape)

X.columns[selector.get_support(indices=True)]

# 1st way to get the list
vector_names = list(X.columns[selector.get_support(indices=True)])
print(vector_names)

#QUESTA è INFO-GAIN
selector2 = SelectKBest(mutual_info_classif, k=3)#QUESTO CI DA LE PRIME K FEATURE IN ORDINE DI IMPORTANZA (O ALMENO QUESTA è L'IPOTESI)
selector2.fit(X,y)
X_new2 = selector2.transform(X)
print(X_new2.shape)
X.columns[selector2.get_support(indices=True)]
vector_names2=list(X.columns[selector2.get_support(indices=True)])
print(vector_names2)



