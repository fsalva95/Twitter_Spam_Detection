import os
import shutil
import sys  ########
from sklearn.model_selection import train_test_split
import shutil
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf  #per gpu(da togliere se non vi interessa 

filename = "data_train.txt"


X = []  #intero dataset
Y = []  #groundtruth

train_x = []  #attributi learning
train_y = []  #ground truth
test_x = []
test_y = []


with tf.device('/gpu:0'): #usato per la gpu (se non volete usare la gpu potete togliere questo comando

        # reading csv file
        with open(filename, 'r',encoding="utf8") as file:

                i=0
                for row in file:
                        if i!=0 and len(row.split(";;$;;"))==12 and i<500:# i<500 per vedere che il codice funziona (è da togliere per trainare tutto il db
                                value=row.split(";;$;;")
                                classification=value[4]
                                value=list(map(str.strip,value))
                                Y.append(classification)

                                value.remove(value[4])  #rimuovo classificazione
                                value=list(map(float,value))
                                X.append(value)
                                print(value)
                                print(row)
                        i=i+1


        #print(X) #troppo pesante
        print("FATTO")
        #print(Y)

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.20)
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(train_x, train_y)
        print("FATTO")
        y_pred = svclassifier.predict(test_x)
        print("FATTO")

        print(confusion_matrix(test_y,y_pred))
        print(classification_report(test_y,y_pred))



