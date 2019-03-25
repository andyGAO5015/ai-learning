# -*-coding:utf-8-*-
# 读数据，这里使用一个简单数据集
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 生成单个物品项集
def createC1(dataSet):
    C1 = []
    # 循环查看每一条交易记录
    # 对每一条交易记录，判断记录中的物品是否已经在C1中
    # 如果不在则添加
    # 使用C1.append([XXX])，注意要使用中括号
    # 【代码待补全】
    for transaction in dataSet:
        for item  in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # 返回创建好的项集
    # 这里使用了frozenset构建不可变集合
    return list(map(frozenset, C1))


# 对数据集中的每条交易记录tran：
# 对每个候选项集can：
# 检查一下can是否是tran的子集：
# 如果是则增加can的计数值
# 对每个候选项集：
# 如果其支持度不低于最小支持度，则保留该项集
# 返回频繁项集列表
# 根据伪代码和提示，补全代码。

#将候选集Ck转换为频繁项集Lk
#D：原始数据集
#Cn: 候选集项Ck
#minSupport:支持度的最小值
def scanD(D, Ck, minSupport):
    # 创建一个字典保存各个候选集的计数
    ssCnt = {}
    for tid in D:
        for can in Ck:
            # 判断can是否为tid子集，可使用issubset()函数
            # 如果是则增加can的计数，保存在ssCnt中
            # 更新计数时要先判断can是否已经在ssCnt的keys中
            # 【代码待补全】
            if can.issubset(tid):
                if can not in ssCnt.keys():ssCnt[can]=1
                else:ssCnt[can]+=1

    # 获取总的交易数
    numItems = float(len(D))
    # 创建返回列表保存频繁项集
    retList = []
    # 创建支持度字典保存各个项集的支持度，以便后续计算置信度
    supportData = {}
    for key in ssCnt:
    # 计算项集的支持度，使用该项集的计数除以总的交易数即可
    # 如果支持度大于最小支持度，则将项集即key保存到retList中
    # 同时将支持度的值保存到supportData中
    # 【代码待补全】
        support=ssCnt[key]/numItems
        if support>=minSupport:
            retList.append(key)
        supportData[key]=support
    # 返回频繁项集列表和支持度字典
    return retList, supportData


# 构建一个由k个项组成的候选集列表。这里的Lk是已知的每一项包
# 含k - 1
# 个物品的频繁项集列表。
def createCk(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 获取两个集合的前k-2个项
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
            # 如果前k-2个项相同，则将两个集合合并
            # 将合并结果添加到返回列表中
            # 添加使用append函数，合并集合使用|
            # 【代码待补全】
                retList.append(Lk[i]|Lk[j])
    return retList

# apriori函数接收两个参数，第一个为数据集，第二个为最小支持度，为可选
# 参数，默认使用0.5
def apriori(dataSet, minSupport=0.5):
    # 生成单个物品项集
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    # 根据单个物品项集获取1-频繁项集及其支持度
    L1, supportData = scanD(D, C1, minSupport)
    # 使用L保存所有的频繁项集
    L = [L1]
    k = 2 # k为项集中每一项物品数
    while len(L[k-2]) > 0:
        #【思考】为什么是k-2
        # 生成k-候选集
        Ck = createCk(L[k-2],k)
        # 调用scanD函数获取k-频繁项集及其支持度
        # 保存到变量Lk和supK中
        #【代码待补全】
        Lk,supK=scanD(D,Ck,minSupport)
        # 更新支持度字典supportData、频繁项集列表L以及k
        # 支持度字典使用update()函数，频繁项集使用append()函数
        #【代码待补全】
        supportData.update(supK)
        L.append(Lk)
        k+=1
    return L, supportData


#生成关联规则
# L: 频繁项集列表
#supportData: 包含频繁项集支持数据的字典
#minConf 最小置信度
def generateRules(L,supportData,minConf=0.7):
    # 包含置信度的规则列表
    bigRuleList=[]
    # 从频繁二项集开始遍历
    for i in range(1,len(L)):
        #【思考】这里为什么是从1开始
        # 注意这里的i是(i+1)-频繁项集的下标
        for freqSet in L[i]:
            # 构造只包含单个元素集合的列表
            H1 = [frozenset([item]) for item in freqSet]
            # 如果频繁项集中的元素数目超过2(通过i判断即可)
            # 则调用规则合并函数
            # 否则调用置信度计算函数
            # 函数签名在接下来的第②和第③步给出
            # 提示：这里均不需要使用到返回值
            # 【代码待补全】
            if(i>1):
                mergeRules(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

# 计算是否满足最小可信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    # 创建修剪后的规则列表H
    prunedH = []
    for conseq in H:
        # 计算以conseq作为右件的规则的置信度conf
        # 使用supportData中的数据即可
        # 提示：conseq作为右件时，左件为freqSet-conseq
        # 左右件同时出现的支持度刚好等于freqSet的支持度
        # 【代码待补全】
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        # 如果置信度满足最小置信度要求，则打印规则
        # 并将规则添加到brl中，同时将右件添加到修剪后的规则列表中
        if conf >= minConf:
            # 打印规则
            print(freqSet - conseq,'-->', conseq,'conf:', conf)
            # 添加规则
            # 使用append函数，添加内容为(左件,右件,置信度)
            # 括号也要
            # 【代码待补全】
            # 元组中的三个元素：前件、后件、置信度
            brl.append((freqSet - conseq, conseq, conf))
            # 添加右件到修剪后的规则列表中，用append函数
            # 【代码待补全】
            prunedH.append(conseq)
    # 返回后件列表
    return prunedH

# 对规则进行评估
# freqSet：频繁项集
# H：可以出现在规则右部的元素列表H
def mergeRules(freqSet, H, supportData, brl, minConf):
    # 获取目前规则列表中规则的右件的元素个数
    m = len(H[0])

    # 确保该频繁项集大到可以移除大小为m的子集
    if len(freqSet) > (m + 1):
        # 调用createCk函数来生成H中元素的无重复组合
        Hmp1 = createCk(H, m + 1)
        # 调用calcConf函数修剪规则列表
        # 返回值保存到Hmp1
        # 【代码待补全】
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # 如果修剪后的规则列表中包含多条规则，则继续尝试合并
        if len(Hmp1) > 1:
            mergeRules(freqSet, Hmp1, supportData, brl, minConf)

# 添加一个测试函数。
def test(minSupport=0.5, minConf=0.7):
        data = loadDataSet()
        L, supportData = apriori(data, minSupport)
        print("支持度不小于：",minSupport,"频繁项集：")
        print(L)
        print("频繁项集字典：")
        print(supportData)
        print("置信度不小于：",minConf,"的关联规则：")
        rules = generateRules(L, supportData, minConf)
        return rules

if __name__=='__main__':
    test()