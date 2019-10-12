from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classity0(inX, dataset, labels, k):
    datasetSize = dataset.shape[0];
    # 距离计算
    diffMat = tile(inX, (datasetSize, 1)) - dataset
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDis = distances.argsort()  # sort index
    # print(sortedDis)
    # 统计标签出现频次
    classCount = {}
    for i in range(k):
        label = labels[sortedDis[i]];
        classCount[label] = classCount.get(label, 0) + 1
    # 统计排序
    sortedCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    lenLines = len(arrayLines)
    returnMatrix = zeros((lenLines, 3))
    labels = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        lineList = line.split('\t')
        returnMatrix[index, :] = lineList[0:3]
        labels.append(int(lineList[-1]))
        index += 1
    fr.close()
    return returnMatrix, labels


# 数值归一化，newValue=(oldValue-min)/(max-min)
def autoNorm(dataset):
    min = dataset.min(0)
    max = dataset.max(0)
    ranges = max - min
    normDataset = zeros(shape(dataset))
    m = dataset.shape[0]
    normDataset = dataset - tile(min, (m, 1))
    normDataset = normDataset / tile(ranges, (m, 1))
    return normDataset, ranges, min


def datingClassTest():
    ratio = 0.1
    dataset, labels = file2matrix("datas/dataset.txt")
    normDataset, ranges, min = autoNorm(dataset)
    m = normDataset.shape[0]
    numTestVecs = int(m * ratio)
    errorCount = 0
    for i in range(numTestVecs):
        result = classity0(normDataset[i, :], normDataset[numTestVecs:m, :], labels[numTestVecs:m], 3)
        print('classifier came back with:{}, the real answer is:{}'.format(result, labels[i]))
        if (result != labels[i]):
            errorCount += 1

    print('the total error rate is:{}'.format(errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small does', 'in large does']
    percebtTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent filter miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    dataset, labels = file2matrix("datas/dataset.txt")
    normMat, ranges, minValue = autoNorm(dataset)
    inArr = array([ffMiles, percebtTats, iceCream])
    result = classity0(inArr, normMat, labels, 3)
    print('you will probably like this person:', resultList[result - 1])
