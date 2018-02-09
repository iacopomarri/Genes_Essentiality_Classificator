from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from Functions import setSplit
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


#iterando sulla dimensione del training set (preso da Cerevisiae) e sulla quantià di elementi essenziali e non, presenti
# in esso, si effettuano vari test di predizione, si termina quando sene trova uno con almeno la precisione desiderata, e viene
# ritornato l'algoritmo fittato con il tranining set in questione.
# potrebbe non trovare un training set con tale precisione.
def test(estimator, Cerevisiae, essNstart, essNend, essNpass, sizeRangeStart, sizeRangeend, sizeRangepass):
    max=0
    maxsize=0
    maxessN=0
    maxAccuracy=0
    bestMatrix=0
    bestEstimator=0

    for essN in range(essNstart, essNend, essNpass):
        for size in range(sizeRangeStart, sizeRangeend, sizeRangepass):
            # funzione che splitta Cerevisiae in trainset e testset. permette di scegliere la dimensione del trainset(size) e la quantità
            # di elementi essenziali al suo interno (essN). Divide entrambi i set in X ed y. 3500 elementi tot in Cerevisiae
            X_train, y_train, X_test, y_test = setSplit(size, essN, copy.deepcopy(Cerevisiae))
            y_pred = estimator.fit(X_train, y_train).predict(X_test)
            accuracy = 100 * accuracy_score(y_test, y_pred)
            matrix = confusion_matrix(y_test, y_pred)


            falseFactor = matrix[0][0] / (matrix[0][0] + matrix[0][1])
            trueFactor = matrix[1][1] / (matrix[1][1] + matrix[1][0])
            avarage=(falseFactor+trueFactor)/2

            # stampa solo i dati dei parametri per cui il perceptron è corretto almeno ad un certo livello indicato
            if falseFactor > 0.55 and trueFactor > 0.55:
                if avarage>max:
                    max=avarage
                    maxsize = size
                    maxessN = essN
                    maxAccuracy = accuracy
                    bestMatrix = matrix
                    bestEstimator=copy.deepcopy(estimator)

                    X_train_best = copy.deepcopy(X_train)
                    y_train_best=copy.deepcopy(y_train)
                    X_test_best=copy.deepcopy(X_test)
                    y_test_best=copy.deepcopy(y_test)
                    y_pred_best=copy.deepcopy(y_pred)


                print(" ")
                print(" ")
                print("Numero di elementi essenziali nel training set: ", end="")
                print(essN)
                print("Dimensione del training set: ",end="")
                print(size)
                print("Accuratezza: ",end="")
                print(accuracy)
                print("Matrice di confusione: ")
                print(matrix)
                print(classification_report(y_test, y_pred))
        print(essN)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_best, y_pred_best)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print(roc_auc)
    plt.title('Receiver Operating charateristics')
    plt.plot(false_positive_rate, true_positive_rate, 'b')
    label = ('AUC=%0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False positive Rate')
    return bestEstimator, max, maxsize, maxessN, maxAccuracy, bestMatrix, X_train_best, y_train_best, X_test_best, y_test_best, y_pred_best, plt