#-*- coding:utf-8 -*-
from numpy import *
import os
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# def kNNClassify(newInput, dataSet, labels, k):
#     ##Step1: 计算新数据到训练样本集中各个数据的欧式距离。
#     # 这里可以使用tile(A, reps)函数，它可以把A重复reps次；
#     # 当reps为(m,n)时，表示垂直方向上重复m次，水平方向上重复n次；
#     # 当reps只有一个数时，表示的是水平方向重复的次数；
#     # python里直接用－表示两个矩阵的对应元素相减;
#     # x的n次方用x**n表示;
#     # 用sum(A, axis=0)函数进行矩阵的按列求和或者按行求和，当axis=0
#     # 时表示按列求和，axis=1时表示按行求和。
#     #【代码待补全】
#
#     ##Step2: 对得到的距离排序
#     # 可使用argsort(A)函数，该函数的返回值为按照升序排序的下标。
#     # 使用变量sortedDistIndices保存排序结果。
#     # 【代码待补全】
#
# 	# 创建一个字典来统计前k个最小距离中各个类别出现的次数
#     classCount={}
#     for i in range(k):
#     ##Step3: 获取前k个最小距离的分类标签
#         voteLabel = labels[sortedDistIndices[i]]
#
# ##Step4: 统计该分类标签在这k个最相似数据中出现的次数
#         			# 当键值voteLabel不存在时，get()函数返回0
#     classCount[voteLabel] = classCount.get(voteLabel,0)+1
#
# ##Step5:选择k个最相似数据中出现次数最多的分类，把分类结果保存
# # 在predictedClass中并返回
# 		# 可用classCount.items()获得该字典的内容
#     		# 【代码待补全】
#
#     return predictedClass

# classify using kNN
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row

    ## step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy numSamples rows for dataSet
    diff = tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    ## step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = argsort(distance)

    classCount = {}  # define a dictionary (can be append element)
    for i in range(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        ## step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex


# def img2vector(filename):
#     # 初始化返回向量returnVect
#     # 【代码待补全】
#
#     # 打开文件
#     fileIn = open(filename)
#     for i in range(32):
#         # 读取整行数据
#         lineStr = fileIn.readline()
#         for j in range(32):
#     # 将头32个字符保存在返回变量中
#     # 注意使用int()函数将数据保存成数值
#     # 【代码待补全】
#
#
# # 关闭文件
# fileIn.close()
# return returnVect
# convert image to vector
def img2vector(filename):
    rows = 32
    cols = 32
    imgVector = zeros((1, rows * cols))
    fileIn = open(filename)
    for row in range(rows):
        lineStr = fileIn.readline()
        for col in range(cols):
            imgVector[0, row * 32 + col] = int(lineStr[col])

    return imgVector
# ②编写函数，读取训练集和测试集。按照提示补全代码。
#
# def loadDataSet():
#     ##Step1: 获取训练集
#     print("---Getting training set...")
#
#
# # 根据实际情况设置数据集路径
# dataSetDir = "path to your dataset"
# trainingFileList = os.listdir(dataSetDir + "trainingDigits")
# numSamples = len(trainingFileList)
#
# train_x = zeros((numSamples, 1024))
# train_y = []  # 保存标签数据，即分类结果
# for i in range(numSamples):
#     filename = trainingFileList[i]
# train_x[i, :] = \
#     img2vector(dataSetDir + "trainingDigits/%s" % filename)
# label = int(filename.split('_')[0])
# train_y.append(label)
#
# ##Step2: 获取测试集
# # 仿照训练集的获取代码，编写获取测试集的逻辑
# # 要求使用test_x保存数据，test_y保存类别
# # 【代码待补全】
# return train_x, train_y, test_x, test_y
#
# load dataSet
def loadDataSet():
    ## step 1: Getting training set
    print("---Getting training set...")

    dataSetDir = 'kNNdata/'
    trainingFileList = os.listdir(dataSetDir + 'trainingDigits')  # load the training set
    numSamples = len(trainingFileList)

    train_x = zeros((numSamples, 1024))
    train_y = []
    for i in range(numSamples):
        filename = trainingFileList[i]

        # get train_x
        train_x[i, :] = img2vector(dataSetDir + 'trainingDigits/%s' % filename)

        # get label from file name such as "1_18.txt"
        label = int(filename.split('_')[0])  # return 1
        train_y.append(label)

        ## step 2: Getting testing set
    print("---Getting testing set...")

    testingFileList = os.listdir(dataSetDir + 'testDigits')  # load the testing set
    numSamples = len(testingFileList)
    test_x = zeros((numSamples, 1024))
    test_y = []
    for i in range(numSamples):
        filename = testingFileList[i]

        # get train_x
        test_x[i, :] = img2vector(dataSetDir + 'testDigits/%s' % filename)

        # get label from file name such as "1_18.txt"
        label = int(filename.split('_')[0])  # return 1
        test_y.append(label)

    return train_x, train_y, test_x, test_y
# ③编写测试函数。
#
# def testHandWritingClass():
#
#
# # 获取数据集
# print("Step 1: Load data...")
# # 要求使用train_x, train_y, test_x, test_y作为变量
# # 【代码待补全】
#
# # 由于knn是不需要训练步骤的，所以这里直接使用pass跳过
# print("Step 2: Training...")
# pass
#
# print("Step 3: Testing...")
# numTestSamples = test_x.shape[0]
# matchCount = 0  # 用以统计分类正确的数目
# for i in range(numTestSamples):
# # 对测试集中的数据进行分类，取k=3，将得到的结果与标签对
# # 比，如果相等则分类正确的数目加一
# # 【代码待补全】
#
# # 所有测试集数据都跑完后计算分类的正确率，保存到acuuracy变量
# # 【代码待补全】
#
# # 显示结果
# print("Step 4: Show the result...")
# print("The classify accuracy is %.2f%%" % (accuracy * 100))
# test hand writing class
def testHandWritingClass():
    ## step 1: load data
    print("step 1: load data...")

    train_x, train_y, test_x, test_y = loadDataSet()

    ## step 2: training...
    print("step 2: training...")

    pass

    ## step 3: testing
    print("step 3: testing...")

    numTestSamples = test_x.shape[0]
    matchCount = 0
    for i in range(numTestSamples):
        predict = kNNClassify(test_x[i], train_x, train_y, 3)
        if predict == test_y[i]:
            matchCount += 1
    accuracy = float(matchCount) / numTestSamples

    ## step 4: show the result
    print("step 4: show the result...")

    print('The classify accuracy is: %.2f%%' % (accuracy * 100))





if __name__=='__main__':
    group, labels = createDataSet()
    a1 = kNNClassify([0, 0], group, labels, 3)
    print("simple classify result:" + a1);
    testHandWritingClass()