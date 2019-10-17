from math import log


def create_dataset():
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'n0'],
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
