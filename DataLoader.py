import csv
import copy


def loadCerevisiae():
    # load Cerevisiae dataset
    with open('cerevisiae_ALL_noNaN.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        Cerevisiae = list(spamreader)

    # exctract from Cerevisiae unneeded features (attributes)
    for i in range(len(Cerevisiae)):
        Cerevisiae[i].pop(0)  # identifier
        Cerevisiae[i].pop(21)
        Cerevisiae[i].pop(20)
        Cerevisiae[i].pop(16)
        Cerevisiae[i].pop(15)
        Cerevisiae[i].pop(14)
        Cerevisiae[i].pop(13)
        Cerevisiae[i].pop(12)
        Cerevisiae[i].pop(11)
        for j in range(len(Cerevisiae[i])):
            Cerevisiae[i][j] = float(Cerevisiae[i][j])  # casting to float each attribute

    return Cerevisiae

def loadMikatae():
    # load Mikatae dataset
    with open('SMikatae-noNaN-DATA.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        Mikatae = list(spamreader)

    # use a temp list to clear uncomplete rows from Mikatae
    temp = []
    flag = False
    for j in range(len(Mikatae)):
        for i in range(len(Mikatae[j])):
            if (Mikatae[j][i] == ''):
                flag = True
        if flag == False:
            temp.append(Mikatae[j])
        flag = False
    Mikatae = copy.deepcopy(temp)
    del (temp)
    del (flag)

    for i in range(len(Mikatae)):
        for j in range(len(Mikatae[i])):
            Mikatae[i][j] = float(Mikatae[i][j])  # casting to float each attribute

    return Mikatae