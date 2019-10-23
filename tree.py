# 决策树
from math import log
import operator
import treePlotter


def create_dataset():
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flipper']
    return dataset, labels


# 计算香农熵
def calc_shannonent(dataset):
    num = len(dataset)
    labelcount = {}
    # 统计次数
    for vec in dataset:
        label = vec[-1]
        if label not in labelcount.keys():
            labelcount[label] = 0
        labelcount[label] += 1
    shannonent = 0
    for key in labelcount:
        prob = float(labelcount[key] / num)
        shannonent -= prob * log(prob, 2)
    return shannonent


# 划分数据集
def split_dataset(dataset, axis, value):
    retdataset = []
    for vec in dataset:
        if vec[axis] == value:
            featvec = vec[:axis]
            featvec.extend(vec[axis + 1:])
            retdataset.append(featvec)
    return retdataset


def choose_best_feature(dataset):
    numfeature = len(dataset[0]) - 1
    baseEntropy = calc_shannonent(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历feature
    for i in range(numfeature):
        featList = [e[i] for e in dataset]
        uniqueVals = set(featList)  # 当前特征的种类
        newEntropy = 0.0
        for val in uniqueVals:
            subDataset = split_dataset(dataset, i, val)
            prob = len(subDataset) / float(len(dataset))
            newEntropy += prob * calc_shannonent(subDataset)
        print("第{}个特征香农熵为：{}".format(i, newEntropy))
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedCount = sorted(classCount.items(), operator.itemgetter(1), True)
    return sortedCount[0][0]


def createTree(dataset, labels):
    # 类别相同,停止递归
    classlist = [x[-1] for x in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    # 划分完所有特征
    if len(dataset[0]) == 1:
        return majorityCnt(classlist)

    best_feat = choose_best_feature(dataset)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del (labels[best_feat])
    feat_val = [x[best_feat] for x in dataset]
    unique_val = set(feat_val)
    for val in unique_val:
        sublabels = labels[:]
        my_tree[best_feat_label][val] = createTree(split_dataset(dataset, best_feat, val), sublabels)

    return my_tree


def classify(tree, featlabels, testvec):
    firstLable = list(tree.keys())[0]
    seconddic = tree[firstLable]
    featIndex = featlabels.index(firstLable)
    for key in seconddic.keys():
        if key == testvec[featIndex]:
            if type(seconddic[key]).__name__ == 'dict':
                classLabel = classify(seconddic[key], featlabels, testvec)
            else:
                classLabel = seconddic[key]

    return classLabel


if __name__ == '__main__':
    dataset, labels = create_dataset()
    print(dataset)
    #print(calc_shannonent(dataset))

    # dataset[0][-1] = 'maybe'
    # print(dataset)
    # print(calc_shannonent(dataset))

    # print(split_dataset(dataset, 0, 1))
    # print(split_dataset(dataset, 0, 0))
    #print(choose_best_feature(dataset))

    tree = createTree(dataset, labels.copy())
    print(tree)

    # treePlotter.createPlot(tree)
    print(classify(tree, labels, [1, 1]))
    print(classify(tree, labels, [0, 1]))

    # 隐性眼镜
    fr = open('datas/lenses.txt')
    lenses = [x.strip().split('\t') for x in fr.readlines()]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensestree=createTree(lenses,labels)
    print(lensestree)
    treePlotter.createPlot(lensestree)