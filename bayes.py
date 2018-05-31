# -*-coding:utf-8-*-
import numpy as np

import math
import re

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1表示侮辱性言论，0表示正常言论
    return postingList, classVec

#构建词汇表生成函数
def createVocabList(dataSet):
        vocabSet=set([])
        for document in dataSet:
                vocabSet=vocabSet|set(document) #取两个集合的并集
        return list(vocabSet)
#将输入语句转化为单词频率向量，表示单词表中的哪些单词出现过
#构建词向量。这里采用的是词集模型，即只需记录每个词是否出现，而不考虑其出现的次数。需要记录词出现的次数的叫词袋模型。
def setOfWords2Vec(vocabList, inputSet):
    returnVec = np.zeros(len(vocabList))  # 生成零向量的array
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  # 单词出现则记为1
        else:
            print('the word:%s is not in my Vocabulary!' % word)
    return returnVec  # 返回全为0和1的向量

#分类器函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print(trainMat)
    p0V, p1V, pAb = trainNB(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

#朴素贝叶斯训练函数，输入为全部文档的单词频率向量集合。类标签向量
def trainNB(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #文档数量
    numWords = len(trainMatrix[0]) #单词表长度
    pAbusive = sum(trainCategory)/float(numTrainDocs) #侮辱性词语的频率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords) #分子初始化为1
    p0Denom = 2.0; p1Denom = 2.0                   #分母初始化为2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1: #假设是侮辱性语句
            p1Num += trainMatrix[i] #矢量相加，将侮辱性语句中出现的词语频率全部加1
            p1Denom += sum(trainMatrix[i]) #屈辱性词语的总量也添加
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom #对每一个元素做除法
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive #返回全部词语非侮辱性词语中的频率。全部词语在侮辱性词语中的频率。侮辱性语句的频率

if __name__=='__main__':
    testingNB()