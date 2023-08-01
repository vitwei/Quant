# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:48:02 2023

@author: jczeng
"""
import numpy as np
import pandas as pd
import math
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import chi2
from sklearn.metrics import roc_curve
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt 




def decision_tree_binning(data,var,target,max_bin=5):
    clf = DecisionTreeClassifier(
        criterion='entropy',  # “信息熵”最小化准则划分
        max_leaf_nodes=max_bin,  # 最大叶子节点数
        min_samples_leaf=0.05)  # 叶子节点样本数量最小占比
    x=data[var].array
    y=data[target]
    clf.fit(x.reshape(-1, 1), y)  # 训练决策树
    # 根据决策树进行分箱
    n_nodes = clf.tree_.node_count  # 决策树节点
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold 
    # 开始分箱
    boundary = []
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])
    boundary.sort()
    min_x = x.min()-0.1
    max_x = x.max()+0.1
    # max_x = x_value.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x]
    return boundary
def markbining(data,var,boundary):
    for mark in range(len(boundary)-1):
        a=boundary[mark]
        b=boundary[mark+1]
        data.loc[(data[var]<b)&(data[var]>a),['mark']]=mark
def goodandbad_count(data,target,markidx,markbin):
    ''' 计算iv值
    Args:
        data: DataFrame，拟操作的数据集
        target: String，Y列名称
        markidx:int,分箱总数
    Returns:
        IV值， float
    '''
    reslist = []
    totalgoodsum=data.loc[data[target]==0].shape[0]
    totalbadsum=data.loc[data[target]!=0].shape[0]
    iv=0
    for markvalue in markbin:
        data_bad = data.loc[(data[target]!=0)&(data[markidx]==markvalue)].shape[0]
        data_good = data.loc[(data[target]==0)&(data[markidx]==markvalue)].shape[0]
        pctgood=data_good/totalgoodsum
        pctbad=data_bad/totalbadsum
        if pctgood==0 or pctbad==0:
            print('分箱')
            print(markvalue)
            print('有0')
            break
        woe=math.log(pctbad/pctgood)
        littleiv=(pctbad-pctgood)*woe
        reslist.append([pctbad,pctgood,woe,littleiv])
        iv+=littleiv
    reslist.append(iv)
    return reslist     
def calculate_chi(freq_array):
    """ 计算卡方值
    Args:
        freq_array: Array，待计算卡方值的二维数组，频数统计结果
    Returns:
        卡方值，float
    """
    # 检查是否为二维数组
    assert(freq_array.ndim==2)
    
    # 计算每列的频数之和
    col_nums = freq_array.sum(axis=0)
    # 计算每行的频数之和
    row_nums = freq_array.sum(axis=1)
    # 计算总频数
    nums = freq_array.sum()
    # 计算期望频数
    E_nums = np.ones(freq_array.shape) * col_nums / nums
    E_nums = (E_nums.T * row_nums).T
    # 计算卡方值
    tmp_v = (freq_array - E_nums)**2 / E_nums
    # 如果期望频数为0，则计算结果记为0
    tmp_v[E_nums==0] = 0
    chi_v = tmp_v.sum()
    return chi_v
def get_chimerge_bincut(data, var, target, max_group=None, chi_threshold=None):
    """ 计算卡方分箱的最优分箱点
    Args:
        data: DataFrame，待计算卡方分箱最优切分点列表的数据集
        var: 待计算的连续型变量名称
        target: 待计算的目标列Y的名称
        max_group: 最大的分箱数量（因为卡方分箱实际上是合并箱体的过程，需要限制下最大可以保留的分箱数量）
        chi_threshold: 卡方阈值，如果没有指定max_group，我们默认选择类别数量-1，置信度95%来设置阈值
        如果不知道卡方阈值怎么取，可以生成卡方表来看看，代码如下：  
        import pandas as pd
        import numpy as np
        from scipy.stats import chi2
        p = [0.995, 0.99, 0.975, 0.95, 0.9, 0.5, 0.1, 0.05, 0.025, 0.01, 0.005]
        pd.DataFrame(np.array([chi2.isf(p, df=i) for i in range(1,10)]), columns=p, index=list(range(1,10)))
    Returns:
        最优切分点列表，List
    """
    freq_df = pd.crosstab(index=data[var], columns=data[target])
    # 转化为二维数组
    freq_array = freq_df.values
    
    # 初始化箱体，每个元素单独一组
    best_bincut = freq_df.index.values
    
    fvalue=[]
    fidx=[]
    # 初始化阈值 chi_threshold，如果没有指定 chi_threshold，则默认选择target数量-1，置信度95%来设置阈值
    if max_group is None:
        if chi_threshold is None:
            chi_threshold = chi2.isf(0.05, df = freq_array.shape[-1])
    
    # 开始迭代
    while True:
        min_chi = None
        min_idx = None
        fvalue=[]
        fidx=[]
        for i in range(len(freq_array) - 1):
            # 两两计算相邻两组的卡方值，得到最小卡方值的两组
            v = calculate_chi(freq_array[i: i+2])
            fvalue.append(v)
            fidx.append(i)
        tmp=pd.DataFrame(columns=['chi','idx'])
        tmp['chi']=fvalue
        tmp['idx']=fidx
        tmp.sort_values(by='chi',inplace=True)
        min_chi =tmp.iat[0,0]
        # 是否继续迭代条件判断
        # 条件1：当前箱体数仍大于 最大分箱数量阈值
        # 条件2：当前最小卡方值仍小于制定卡方阈值
        if (max_group is not None and max_group < len(freq_array)) or (chi_threshold is not None and min_chi < chi_threshold):
            tempkey=(len(freq_array)-max_group)#还需要分的
            fvlist=tmp.loc[tmp['chi']==min_chi,['idx']]
            tmplist=fvlist.index.tolist()
            if len(tmplist)<tempkey:
                for fv in tmplist:
                    if fv+1>len(freq_array) - 1:
                        continue
                    newres = freq_array[fv] + freq_array[fv+1]
                    freq_array[fv] = newres
                    freq_array = np.delete(freq_array, fv+1, 0)
                    best_bincut = np.delete(best_bincut, fv+1, 0)
            else:
                tm1=tmplist[:tempkey]
                for fv in tm1:
                    if fv+1>len(freq_array) - 1:
                        continue
                    newres = freq_array[fv] + freq_array[fv+1]
                    freq_array[fv] = newres
                    freq_array = np.delete(freq_array, fv+1, 0)
                    best_bincut = np.delete(best_bincut, fv+1, 0)
        else:
            break
    
    # 把切分点补上头尾
    best_bincut = best_bincut.tolist()
    best_bincut.append(data[var].min())
    best_bincut.append(data[var].max())
    best_bincut_set = set(best_bincut)
    best_bincut = list(best_bincut_set)
    
    best_bincut.remove(data[var].min())
    best_bincut.append(data[var].min()-1)
    # 排序切分点
    best_bincut.sort()
    
    return best_bincut
def get_maxks_split_point(data, var, target, min_sample=0.05):  
    """ 计算KS值
    Args:
        data: DataFrame，待计算卡方分箱最优切分点列表的数据集
        var: 待计算的连续型变量名称
        target: 待计算的目标列Y的名称
        min_sample: int，分箱的最小数据样本，也就是数据量至少达到多少才需要去分箱，一般作用在开头或者结尾处的分箱点
    Returns:
        ks_v: KS值，float
        BestSplit_Point: 返回本次迭代的最优划分点，float
        BestSplit_Position: 返回最优划分点的位置，最左边为0，最右边为1，float
    """
    if len(data) < min_sample:
        ks_v, BestSplit_Point, BestSplit_Position = 0, -9999, 0.0
    else:
        freq_df = pd.crosstab(index=data[var], columns=data[target])
        freq_array = freq_df.values
        if freq_array.shape[1] == 1: # 如果某一组只有一个枚举值，如0或1，则数组形状会有问题，跳出本次计算
            # tt = np.zeros(freq_array.shape).T
            # freq_array = np.insert(freq_array, 0, values=tt, axis=1)
            ks_v, BestSplit_Point, BestSplit_Position = 0, -99999, 0.0
        else:
            bincut = freq_df.index.values
            tmp = freq_array.cumsum(axis=0)/(np.ones(freq_array.shape) * freq_array.sum(axis=0).T)
            tmp_abs = abs(tmp.T[0] - tmp.T[1])
            ks_v = tmp_abs.max()
            BestSplit_Point = bincut[tmp_abs.tolist().index(ks_v)]
            BestSplit_Position = tmp_abs.tolist().index(ks_v)/max(len(bincut) - 1, 1)
        
    return ks_v, BestSplit_Point, BestSplit_Position
def get_bestks_bincut(data, var, target, leaf_stop_percent=0.2):

    """ 计算最优分箱切分点
    Args:
        data: DataFrame，拟操作的数据集
        var: String，拟分箱的连续型变量名称
        target: String，Y列名称
        leaf_stop_percent: 叶子节点占比，作为停止条件，默认5%
    
    Returns:
        best_bincut: 最优的切分点列表，List
    """
    min_sample = len(data) * leaf_stop_percent
    best_bincut = []
    
    def cutting_data(data, var, target, min_sample, best_bincut):
        ks, split_point, position = get_maxks_split_point(data, var, target, min_sample)
        
        if split_point != -99999:
            best_bincut.append(split_point)
        
        # 根据最优切分点切分数据集，并对切分后的数据集递归计算切分点，直到满足停止条件
        # print("本次分箱的值域范围为{0} ~ {1}".format(data[var].min(), data[var].max()))
        left = data[data[var] < split_point]
        right = data[data[var] > split_point]
        
        # 当切分后的数据集仍大于最小数据样本要求，则继续切分
        if len(left) >= min_sample and position not in [0.0, 1.0]:
            cutting_data(left, var, target, min_sample, best_bincut)
        else:
            pass
        if len(right) >= min_sample and position not in [0.0, 1.0]:
            cutting_data(right, var, target, min_sample, best_bincut)
        else:
            pass
        return best_bincut
    best_bincut = cutting_data(data, var, target, min_sample, best_bincut)
    
    # 把切分点补上头尾
    best_bincut.append(data[var].min())
    best_bincut.append(data[var].max())
    best_bincut_set = set(best_bincut)
    best_bincut = list(best_bincut_set)
    
    best_bincut.remove(data[var].min())
    best_bincut.append(data[var].min()-1)
    # 排序切分点
    best_bincut.sort() 
    return best_bincut
def get_ks_score(data,var,target):
    goodsum=data.loc[data[target]==0].shape[0]
    badsum=data.loc[data[target]==1].shape[0]
    totalsum=data.shape[0]
    df = pd.crosstab(index=data[var], columns=data[target])
    df['goodpct']=df[0]/goodsum
    df['badpct']=df[1]/badsum
    df['goodsumpct']=df['goodpct'].cumsum()
    df['badsumpct']=df['badpct'].cumsum()
    df['ks']=df['goodsumpct']-df['badsumpct']
    df['ks']=abs(df['ks'])
    plt.plot(df.index,df['goodsumpct'],c='green')
    plt.plot(df.index,df['badsumpct'],c='red')
    plt.plot(df.index,df['ks'],c='blue')
    plt.fill_between(df.index, df['ks'],facecolor="orange",
alpha=0.2) 
    plt.show()
    print(df['ks'].max())    
def doublebining(data,var1,var2,target):
    data['mark']=data[var1]*10+data[var2]
    mark1bin=data[var1].value_counts().index.shape[0]
    mark2bin=data[var2].value_counts().index.shape[0]
    freq_df = pd.crosstab(index=data['mark'], columns=data[target])
    freq_array = freq_df.values
    markidx=np.array(freq_df.index)
    graph={}
    chilist=[]
    marklist=[]
    chi0list=[]
    tmp=[]
    for m1 in range(mark1bin*mark2bin):
      for m2 in range(mark1bin*mark2bin):
        if m1!=m2 and ((m1+1)%mark1bin)!=0 and (m2-m1)==1:
          marklist.append([m1,m2])
        if m1!=m2 and (m2-m1)==mark1bin  :
          marklist.append([m1,m2])
    for row in marklist:
            a=calculate_chi(freq_array[[row[0],row[1]]])
            chilist.append(a)
            if row[0] in graph:
              graph[row[0]].append(row[1])
            else:
              graph[row[0]]=([row[1]])
    df=pd.DataFrame(marklist,columns=['fidx','lidx'])
    df['chi']=chilist
    def BFS(graph,seen,point):
        queue=[]
        queue.append(point)
        minchi=9223372036854775807 #intMaxvalue
        mcbin=[-1,-1]#default bin 
        while(len(queue)>0):
            vertex=queue.pop(0)
            if vertex not in graph:
                continue
            nodes=graph[vertex]
            for w in nodes:
                if w not in seen:
                    seen.add(w)
                    a=df.loc[(df['fidx']==vertex)&(df['lidx']==w),['chi']].iat[0,0]
                    if a< minchi:
                      minchi=a
                      mcbin=[vertex,w]
                      queue.append(w)
                if w in seen:
                    continue
        return [mcbin,minchi]        
    for tmp in list(graph.keys()):
      chi0list.append(BFS(graph,set(),tmp))
    chi0list=pd.DataFrame(chi0list) 
    chi0list.drop_duplicates(1,inplace=True)
    chi0list=chi0list.loc[chi0list[1]<1]
    chi0list.sort_values(1,inplace=True)
    data['beforbin']=data['mark']
    keylist=chi0list[0]
    for tmp in keylist:
        data.loc[data['mark']==markidx[tmp[1]],['mark']]=markidx[tmp[0]]
    return chi0list
 
    
    