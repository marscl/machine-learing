# 决策树
from math import log


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
        print("第{}个特征香农熵为：{}".format(i,newEntropy))
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature


dataset, labels = create_dataset()
print(dataset)
print(calc_shannonent(dataset))

# dataset[0][-1] = 'maybe'
# print(dataset)
# print(calc_shannonent(dataset))

# print(split_dataset(dataset, 0, 1))
# print(split_dataset(dataset, 0, 0))
print(choose_best_feature(dataset))
