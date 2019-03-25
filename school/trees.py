# -*-coding:utf-8-*-
from math import log
import operator
import os
# 创建一个简单的数据集。
# 这个数据集根据两个属性来判断一个海洋生物是否属于鱼类，
# 第一个属性是不浮出水面是否可以生存，第二个属性是是否有鳍。数据集中的第三列是分类结果。
def createDataSet():
    dataSet = [[1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

# 计算给定数据集的香农熵
# 熵越高，则混合的数据也越多，随机变量含有的信息越少，变量的不确定性越大
def calcEntropy(dataSet):
    # 获取总的训练数据数
    numEntries = len(dataSet)
    # 创建一个字典统计各个类别的数据量
    labelCounts = {}
    for featVec in dataSet:
        # 使用下标-1获取所属分类
        currentLabel=featVec[-1]
        # 若获得的类别属于新类别，则初始化该类的数据条数为0
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    entropy=0.0
    for key in labelCounts.keys():
        # 计算p(xi)
        prob=float(labelCounts[key])/numEntries
        # 计算熵
        entropy-=prob*log(prob,2)
    return entropy

# 编写函数，实现按照给定特征划分数据集。
# 返回除axis这一列之外的其他数据
# 待划分的数据集、划分数据集的特征、特征的返回值
def splitDataSet(dataSet,axis,value):
    returnDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            # 将符合特征的数据抽取出来
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            returnDataSet.append(reducedFeatVec)
    return returnDataSet

# 实现特征选择函数。遍历整个数据集，循环计算熵和splitDataSet()函数，找到最好的特征划分方式。
# 选择最好的数据集划分方式
# 数据是由列表元素组成的列表，所有的列表元素都要具有相同的数据长度
# 数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签
def chooseBestFeatureToSplit(dataSet):
    # 获取属性个数，保存到变量numFeatures
    # 注意数据集中最后一列是分类结果
    numFeatures=len(dataSet[0])-1
    # 整个数据集的原始香农熵，无序度量值
    baseEntropy = calcEntropy(dataSet)
    bestInfoGain = 0.0;bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        # 获取数据集中某一属性的所有取值
        featList = [example[i] for example in dataSet]
        # 获取该属性所有不重复的取值，保存到uniqueVals中
        uniqueVals=set(featList)
        newEntropy=0.0
        # 遍历当前特征中的所有唯一属性值，对每个特征划分异次数据集
        for value in uniqueVals:
            # 计算每种划分方式的信息熵
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算按照第i列的某一个值分割数据集后的熵
            # 参考文档开始部分介绍的公式
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcEntropy(subDataSet)
        # tableVocabulary[i] = newEntropy
        # 计算按每个数据特征来划分的数据集的熵
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            # 选择最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    # 判断出哪种划分方式得到最大的信息增益，且得出该划分方式所选择的特征
    # 返回最好特征划分的索引值
    return bestFeature

# 决策树创建过程中会采用递归的原则处理数据集。
# 递归的终止条件为：
# 程序遍历完所有划分数据集的属性；或者每一个分支下的所有实例都具有相同的分类。
# 如果数据集已经处理了所有属性，但是类标签依然不是唯一的，
# 此时我们需要决定如何定义该叶子节点，在这种情况下，通常会采用多数表决的方法决定分类。
# 用于找出出现次数最多的分类名称的函数
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
        sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]
# 创建决策树
def createTree(dataSet,labels):
    # 获取类别列表，类别信息在数据集中的最后一列
    classList=[example[-1] for example in dataSet]
    # 以下两段是递归终止条件
    # 如果数据集中所有数据都属于同一类则停止划分
    # 可以使用classList.count(XXX)函数获得XXX的个数，
    # 然后那这个数和classList的长度进行比较，相等则说明
    # 所有数据都属于同一类，返回该类别即可
    if classList.count(classList[0])==len(classList):
        return classList[0]
    # 如果已经遍历完所有属性则进行投票，调用上一步的函数
    # 注意，按照所有属性分割完数据集后，数据集中会只剩下
    # 一列，这一列是分类结果
    if (len(dataSet[0])==1):
        return majorityCnt(classList)
    # 调用特征选择函数选择最佳分割属性，保存到bestFeat
	# 根据bestFeat获取属性名称，保存到bestFeatLabel中
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    # 初始化决策树，可以先把第一个属性填好
    myTree={bestFeatLabel:{}}
    #删除最佳分离属性的名称以便递归调用
    del(labels[bestFeat])
    # 获取最佳分离属性的所有不重复的取值保存到uniqueVals
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 复制属性名称，以便递归调用
        subLabel=labels[:]
        #递归调用本函数生成决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabel)
    return myTree

# 利用构建好的决策树进行分类
def classify(inputTree,featLabels,testVec):
    # 获取树的第一个节点，即属性名称
    firstStr=list(inputTree.keys())[0]
    # 获取该节点下的值
    secondDict=inputTree[firstStr]
    # 获取该属性名称在原属性名称列表中的下标
    # 保存到变量featIndex中
    # 可使用index(XXX)函数获得XXX的下标
    featIndex=featLabels.index(firstStr)
    # 获取待分类数据中该属性的取值，然后在secondDict
    # 中寻找对应的项的取值
    # 如果得到的是一个字典型的数据，说明在该分支下还
    # 需要进一步比较，因此进行循环调用分类函数；
    # 如果得到的不是字典型数据，说明得到了分类结果
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel

# 存储决策树
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
# 根据文件名读取决策树
def grabTree(filename):
    import pickle
    file=os.path.exists(filename)
    if file == False:
        return None
    fr=open(filename,'rb')
    return pickle.load(fr)

if __name__ == "__main__":
    dataSet,labels=createDataSet()
    entropy=calcEntropy(dataSet)
    print(entropy)
    myTree=grabTree("treesdata/treesCache")
    if myTree ==None:
        # 有个bug:当传进去的是labels时会把'no surfacing'吃掉，所以这里传进去labels[:]
        myTree = createTree(dataSet, labels[:])
        # 缓存生成的决策树、方便下次使用
        storeTree(myTree, "treesdata/treesCache")
    print(myTree)

    # result=classify(myTree,labels,[0,0])
    # print(result)
    for i in range(2):
        for j in range(2):
            result = classify(myTree, labels, [i, j])
            print("[",i,",",j,"]",result)
