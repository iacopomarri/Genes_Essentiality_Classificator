#this file uses the perceptron algoritm by training it on the all Cerevisiae dataset, and testing it on
#the usable rows of Mikatae dataset. Results are then matched juts by them numbers of 1 and 0 with the seringaus
#research, in the written relation.

from Test import test
from Functions import Count
from DataLoader import loadCerevisiae, loadMikatae
import csv
import copy
from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score


#variablesw
Cerevisiae=[]
Mikatae=[]
X_train=[]
y_train=[]
X_test=[]
y_test=[]
y_pred=[]

Cerevisiae=loadCerevisiae()
Mikatae=loadMikatae()

#Mikatae e Cerevisiae sono ora ottimizzati per essere usati come dataset. Cerevisiae contiene una colonna in più di mikatae
#che rappresenta l'essenzialità.

cv=KFold(n_splits=5, random_state=5, shuffle=True)
estimator=Perceptron(max_iter=10)

#test è una funzione che iterando sulla dimensione del training set (preso da Cerevisiae) e sulla quantià di elementi essenziali e non, presenti
# in esso, si effettuano vari test di predizione, si termina quando sene trova uno con almeno la precisione desiderata, e viene
# ritornato l'algoritmo fittato con il tranining set in questione.
# potrebbe non trovare un training set con tale precisione.
#trovato il miglior estimatore si usa per predirre l'essenzialità del dataset Mikatae
essentialityNRange=[50, 449, 100]
sizeRange=[450, 2500, 1000]

#valori di ritorno estimator, max, maxsize, maxessN, maxAccuracy, bestMatrix
estimator, prec, trainSize, essElementsInTrain, accuracy, confusionMatrix =test(estimator, Cerevisiae, 50, 449, 10,
                                                                                                        450, 2500, 10)
if estimator!=0:
    pred=estimator.predict(Mikatae)
    print(pred)
    ones, zeros = Count(pred)
    print(ones, zeros)
else:
    print("Non è stato trovato un estimatore soddisfacente")

