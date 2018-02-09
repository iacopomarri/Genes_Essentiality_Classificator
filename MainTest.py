#this file uses the perceptron algoritm by training it on the all Cerevisiae dataset, and testing it on
#the usable rows of Mikatae dataset. Results are then matched juts by them numbers of 1 and 0 with the seringaus
#research, in the written relation.

from Test import test
from sklearn.metrics import confusion_matrix, accuracy_score
from Functions import Count, SplitX_y
from DataLoader import loadCerevisiae, loadMikatae
import csv
import copy
from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score


#variablesw
Cerevisiae=[]
Mikatae=[]

Cerevisiae=loadCerevisiae()
Mikatae=loadMikatae()

#Mikatae e Cerevisiae sono ora ottimizzati per essere usati come dataset. Cerevisiae contiene una colonna in più di mikatae
#che rappresenta l'essenzialità.

estimator=Perceptron(max_iter=10)

#test è una funzione che iterando sulla dimensione del training set (preso da Cerevisiae) e sulla quantià di elementi essenziali e non, presenti
# in esso, si effettuano vari test di predizione, si termina quando sene trova uno con almeno la precisione desiderata, e viene
# ritornato l'algoritmo fittato con il tranining set in questione.
# potrebbe non trovare un training set con tale precisione.
#trovato il miglior estimatore si usa per predirre l'essenzialità del dataset Mikatae


#valori di ritorno estimator, max, maxsize, maxessN, maxAccuracy, bestMatrix
estimator, prec, trainSize, essElementsInTrain, accuracy, confusionMatrix, X_train_best, y_train_best, X_test_best, y_test_best, y_pred_best, plt=\
                                                                                                test(estimator, Cerevisiae, 50, 449, 10,
                                                                                                                           450, 2500, 10)
if estimator!=0:
    print("L'algoritmo è stato allenato su un train set di dimensione ", end="")
    print(trainSize, end="")
    print(" e con un numero di elementi essenziali pari a ", end="")
    print(essElementsInTrain)
    print("Precisione = ", end="")
    print(prec)
    print("Accuratezza = ", end="")
    print(accuracy)
    print("Matrice confusionale = ")
    print(confusionMatrix)
    print("L'estimatore è stato usato poi per predirre l'essenzialità del dataset Mikatae, ottenendo su 4500 elementi, i seguenti risultati:")

    pred=estimator.predict(Mikatae)
    #print(pred)
    ones, zeros = Count(pred)
    print("Elementi essenziali: ", end="")
    print(ones)
    print("Elementi non essenziali: ", end="")
    print(zeros)

    y_pred = estimator.fit(X_train_best, y_train_best).predict(X_test_best)
    accuracy = 100 * accuracy_score(y_test_best, y_pred)
    matrix = confusion_matrix(y_test_best, y_pred)
    print(accuracy)
    print(matrix)
else:
    print("Non è stato trovato un estimatore soddisfacente")

plt.show()