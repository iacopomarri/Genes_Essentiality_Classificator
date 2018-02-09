import csv
import numpy as np
import random
import copy
from sklearn.datasets import load_digits
y_test=[]
Mikatae=[]
flag=False


with open('cerevisiae_ALL_noNaN.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    Dataset= list(spamreader)


with open('SMikatae-noNaN-DATA.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    A= list(spamreader)

#indice=A.pop(0)

for j in range (len(A)):
    for i in range( len (A[j])):
        if(A[j][i]==''):
            flag=True
    if flag==False:
        Mikatae.append(A[j])
    flag=False


for i in range(len(Mikatae)):

    for j in range(len(Mikatae[i])):
        Mikatae[i][j]=float(Mikatae[i][j])




for i in range (len(Dataset)):
    Dataset[i].pop(0)
    Dataset[i].pop(22)
    Dataset[i].pop(21)
    Dataset[i].pop(20)
    Dataset[i].pop(16)
    Dataset[i].pop(15)
    Dataset[i].pop(14)
    Dataset[i].pop(13)
    Dataset[i].pop(12)
    Dataset[i].pop(11)
    for j in range(len(Dataset[i])):
        Dataset[i][j]=float(Dataset[i][j])

print(Mikatae[514])
print(Dataset[512])

print(len(Dataset[i]))
print(len(Mikatae[i]))

