from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 存储训练集和测试集的域名
domaintrainlist = []
domaintestlist = []


class Domain:
    def __init__(self, _name, _label):
        self.name = _name
        self.label = _label

    def returnData(self):
        s = self.name
        length = len(s)
        num = 0
        for i in range(length):
            if s[i] >= '0' and s[i] <= '9':
                num = num + 1
        return [length, num]

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1


def initTrainData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            domaintrainlist.append(Domain(name, label))


def initTestData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            domaintestlist.append(Domain(name, 0))


def main():
    initTrainData("train.txt")
    initTestData("test.txt")
    fl = open('result.txt', 'w')
    featureMatrix = []
    labelList = []
    for item in domaintrainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    for it in domaintestlist:
        if clf.predict([it.returnData()]) == 0:
            fl.write(it.name + ',notdga' + '\n')
        else:
            fl.write(it.name + ',dga' + '\n')


if __name__ == '__main__':
    main()
