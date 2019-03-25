# -*-coding:utf-8-*-
from numpy import *
import matplotlib.pyplot as plt  ##绘图库

# 导入数据集。
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

#定义距离计算函数。这里使用欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  #


# 编写创建质心函数，为给定数据集创建包含k个随机质心的集合。
def randCent(dataSet, k):
    n = shape(dataSet)[1]  #
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

# dataSet样本点,k 簇的个数
# disMeas距离量度，默认为欧几里得距离
# createCent,初始点的选取
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0] #样本数
    clusterAssment = mat(zeros((m,2))) #m*2的矩阵
    centroids = createCent(dataSet, k) #初始化k个中心
    clusterChanged = True
    executeCount=0
    while clusterChanged:      #当聚类不再变化
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k): #找到最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            # 第1列为所属质心，第2列为距离
            clusterAssment[i,:] = minIndex,minDist**2
        executeCount+=1
        print("执行次数:",executeCount,"所得到的质心")
        # print("")
        print(centroids)

        # 更改质心位置
        for cent in range(k):
            #返回质心为cent的点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            #mean axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
            #重新求质心
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def draw(data,center):
    length=len(center)
    fig=plt.figure
    # 绘制原始数据的散点图
    plt.scatter(data[:,0].tolist(),data[:,1].tolist(),s=25,alpha=0.4)
    # 绘制簇的质心点
    for i in range(length):
        plt.annotate('center',xy=(center[i,0],center[i,1]),xytext=\
        (center[i,0]+1,center[i,1]+1),arrowprops=dict(facecolor='red'))
    plt.show()

if __name__ == "__main__":
    datMat=mat(loadDataSet('kMeansdata/testSet.txt'))
    myCentroids,clustAssing=kMeans(datMat,4)
    print("最终质心为：")
    print(myCentroids)
    print("各个所属的族以及到簇的距离")
    print(clustAssing)
    draw(datMat,myCentroids)

