# Genes_Essentiality_Classificator
Questa applicazione usa l'algoritmo Perceptron applicandolo al problema della classificazione dei geni, usando i datasets di 
S.Cerevisiae e S.Mikatae .
In the main.py file che test() function is called, with some parameters useful to test the perceptron by changing them.
the parameters are: 
Nel file Main.py è chiamata la funzione test(), che sta nel file Test.py, e gli sono passati alcuni parametr utili a testare il perceptron:

-Estimator
-Dataset
-essNstart: numero minimo (minore, iniziale) di elementi essenziali presenti nel training set generato. I test partono da questo valore fino ad essNend
-essNend: numero minimo (maggiore, finale) di elementi essenziali presenti nel training set generato.
-essStep: passo con cui essN varia da essNstart a essNmin
-TrainsizeStart: minima dimensione del trainset.
-TrainsizeEnd: massima dimensione del trainset
-TrainsizeStep: passo con cui Trainsize varia da TrainsizeStart a TrainsizeEnd

test() sceglie la combinazione migliore di parametro provandoli in base alle indicazioni inserite, e valutando ognuna in base 
all'accuratezza de fit() e del predict().

Durante l'esecuzione viene stampato ogni poco essN, che varia da essNstart a essNend, per avere una indicazione della percentuale di completamento
del test. 
i risultati vengono mostrati volta per volta quando vengono trovati (solo i migliori) e alla fine viene mostrato il migliore fra tutti.
Dato questo miglior fitting, viene eseguito il test finale sul set Mikatae e vengono stampati i numeri di geni esseniali e di geni non
essenziali.
