from sklearn.linear_model import Perceptron
import csv
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import copy
from Functions import setSplit, plot_curve, setSplitDue
import random
from random import randint
from sklearn.model_selection import KFold
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

#da 0 a 10, 16-17-18
y_test=[]
Mikatae=[]
flag=False

with open('cerevisiae_ALL_noNaN.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    Dataset= list(spamreader)


with open('SMikatae-noNaN.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    A= list(spamreader)

A.pop(0)

for j in range (len(A)):
    for i in range( len (A[j])):
        if(A[j][i]==''):
            flag=True
    if flag==False:
        Mikatae.append(A[j])
    flag=False


for i in range (len(Dataset)):
    Dataset[i].pop(0)
 #   Dataset[i].pop(22)
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

for i in range(len(Mikatae)):
    Mikatae[i].pop(0)
    for j in range(len(Mikatae[i])):
        Mikatae[i][j]=float(Mikatae[i][j])

ytrain=[]

Xtrain=copy.deepcopy(Dataset)
for i in range(len(Dataset)):
    ytrain.append(Xtrain[i].pop())

ss=KFold(n_splits=5, random_state=5, shuffle=True)
estimator = Perceptron(max_iter=10)

#param_grid = {'max_iter':range(5,50,10)}
#clf = GridSearchCV(estimator, n_jobs=1, cv=ss,  param_grid=param_grid)
#clfconfit = clf.fit(Xtrain, ytrain)
#print (clf.best_score_)
#print (clf.best_params_)
#ypred=clfconfit.predict(Mikatae)

#ypred=estimator.fit(Xtrain, ytrain).predict(Mikatae)


#430
#1540

#390
#1930


#240
#1540
max=0
min=10000

def test(f):
        for prcn in range (50,449,10):
            for size in range(450, 2500, 10):

                X_train,y_train,X_test,y_test= setSplitDue(size, prcn, copy.deepcopy(Dataset))

                y_pred=estimator.fit(X_train,y_train).predict(X_test)


                accuracy=100*accuracy_score(y_test, y_pred)

                matrix=confusion_matrix(y_test, y_pred)
                m=matrix[0][0] + matrix[1][1]-matrix[0][1]-matrix[1][0]
                n=matrix[0][1]+matrix[1][0]

                totElementINtest=3500-size
                falseFactor=matrix[0][0]/(matrix[0][0]+matrix[0][1])
                trueFactor=matrix[1][1]/(matrix[1][1]+matrix[1][0])

                if falseFactor>0.55 and trueFactor>0.55:#matrix[0][0]>500 and matrix[1][1]>50:#m>max:# and n<min:
                    #max=m
                    #min=n
                    print(" ")
                    print(" ")
                # print(falseFactor)
                    #print(trueFactor)
                    print(prcn)
                    print(size)
                    print(accuracy)
                    print(matrix)
                    print(classification_report(y_test, y_pred))
                    print("esco")
                    return estimator


f=False
e=test(f)
pred=e.predict(Mikatae)
print(pred)

c = 0
for i in range(len(pred)):
    if pred[i] == 1:
        c += 1
print("c è: ")
print(c)



#y_pred=estimator.fit(X_train,y_train).predict(X_test)

#false_positive_rate, true_positive_rate,thresholds=roc_curve(y_test, y_pred)
#roc_auc=auc(false_positive_rate, true_positive_rate)
#print(roc_auc)
#plt.title('Receiver Operating charateristics')
#plt.plot(false_positive_rate, true_positive_rate, 'b')
#label= ('AUC=%0.2f' % roc_auc)
#plt.legend(loc='lower right')
#plt.plot([0,1],[0,1],'r--')
#plt.xlim([-0.1,1.2])
#plt.ylim([-0.1,1.2])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False positive Rate')
#plt.show()









#BEST PERC=80  TRAINSIZE=900
#BEST prc=26  trnsz=2550
#BEST best 47  2690

#SCOMMENTA STA ROBA PER AVERE LA CURVA QUASI BELLINA
#a=[]
#for i in range (len(Dataset)):
#    a.append(Dataset[i].pop())
#train_sizes=np.logspace(np.log10(.05), np.log10(1.0), 8)
#trainsize, trainscore, testscore = learning_curve(estimator, Dataset, a, n_jobs=-1,cv=ss,  train_sizes=train_sizes)
#plot_curve(trainsize, trainscore, testscore, "curva")
#plt.show()


#SCOMMENTA STA ROBA PER AVERE LA CURVA BRUTTA DATA DAI 3 BEST INDICATI SOPRA TROVATI COL MEGA CICLO FOR
#X_train, y_train,a,b= setSplit(2690, 47,copy.deepcopy(Dataset) )
#train_sizes=np.logspace(np.log10(.05), np.log10(1.0), 8)
#trainsize, trainscore, testscore = learning_curve(estimator, X_train, y_train, n_jobs=-1,cv=ss,  train_sizes=train_sizes)
#plot_curve(trainsize, trainscore, testscore, "curva")
#plt.show()
