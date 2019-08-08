import os
import shutil
import sys  ########

import shutil

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
                if i!=0:
                        value=row.split(";;$;;")
                        classification=value[4]
                        value=list(map(str.strip,value))
                        Y.append(classification)
                        #value.remove(value[0])  #rimuovo id
                        #value.remove(value[0])  #rimuovo testo
                        value.remove(value[4])  #rimuovo classificazione
                        value=list(map(float,value))
                        X.append(value)
                        print(value)
                        print(row)
                i=i+1


print(X)
print("\n")
print(Y)
