import os
import shutil
import sys  ########
from sklearn.model_selection import train_test_split
import shutil
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt




filename = "data_train.txt"


X = []  #intero dataset
Y = []  #groundtruth

train_x = []  #attributi learning
train_y = []  #ground truth
test_x = []
test_y = []

data_graph_quality = []
data_graph_spam = []

# reading csv file
with open(filename, 'r',encoding="utf8") as file:

        i=0
        for row in file:
                if i!=0 and len(row.split(";;$;;"))==12 :#and i<100:  #i<500 per vedere che il codice funziona (è da togliere per trainare tutto il db
                        value=row.split(";;$;;")
                        classification=value[4]
                        value=list(map(str.strip,value))
                        Y.append(classification)

                        value.remove(value[4])  #rimuovo classificazione
                        value=list(map(float,value))
                        value=np.asarray(value)
                        #print(value.shape)
                        X.append(value)
                        if classification == 'Quality':
                            data_graph_quality.append(value)
                        else:
                            data_graph_spam.append(value)
                        #print(value)
                        #print(row)
                i=i+1


#print(X) #troppo pesante
print("FATTO")
#print(Y)
X=np.asarray(X)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.20)
svclassifier = SVC(kernel='rbf', gamma='auto')
print(train_x.shape)
model=svclassifier.fit(train_x, train_y)
print("FATTO")
y_pred = svclassifier.predict(test_x)
print("FATTO")

print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))


data_graph_spam = np.asarray(data_graph_spam)
data_graph_quality = np.asarray(data_graph_quality)
testColumns = ['following','followers','actions', 'is_retweet', 'URLCounted', 'HashtagCounted', 'MensionCounted', 'averageHashtag', 'averageURL', 'wordsCounted', 'SpamWordsCounted']


num_bins = 20
for i in range(len(testColumns)) :
    fig, single_plot = plt.subplots()
    fig.canvas.set_window_title(testColumns[i])
    sorted_data = np.sort(data_graph_spam[:,i])

    yvals=np.arange(len(sorted_data))/float(len(sorted_data)-1)
    
    single_plot.plot(sorted_data,yvals, color='blue') #blue is for spam
    sorted_data = np.sort(data_graph_quality[:,i])

    yvals=np.arange(len(sorted_data))/float(len(sorted_data)-1)
    
    single_plot.plot(sorted_data,yvals,color='red') #red is for quality
    plt.ylim(top=1)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    #plt.xlim(right=0)
    single_plot.set_xlabel(testColumns[i])
    single_plot.set_ylabel('CDF')
    plt.show()
    



#print(X);