# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:44:04 2019

@author: Administrator
"""

import numpy as np
import inspect
#从代码中获取变量名
def retrieve_name_ex(var):
    stacks = inspect.stack()
    try: 
        callFunc = stacks[1].function
        code = stacks[2].code_context[0]
        startIndex = code.index(callFunc) 
        startIndex = code.index("(", startIndex + len(callFunc)) + 1  
        endIndex = code.index(")", startIndex) 
        return code[startIndex:endIndex].strip() 
    except:     
        return ""
#打印附带变量名
def sprint(var): 
    print("{} =\n {}".format(retrieve_name_ex(var),var))

def show_array(list_x):
    print('/----show------/')
    sprint(list_x)
    #列表转矩阵
    array_x = np.array(list_x)
    sprint(array_x)
    #维度,这个维度指的是array的维度
    sprint(array_x.ndim)
    #行列
    sprint(array_x.shape)
    #元素个数
    sprint(array_x.size)
    print('/-----end------/\n\n')
    

#show_array([1,2,3])
#show_array([[1,2,3],[4,5,6]])
#show_array([[[1,2],[3,4]],[[5,6],[7,8]]])
    
#array的创建方式
##指定数据类型
###np.int 8 16 32 64 ,np.float 16 32 64
def show_dtype(list_x,dtype):
    array_a_specify_datatype = np.array(list_x,dtype=dtype)
    sprint(array_a_specify_datatype.dtype)
    sprint(array_a_specify_datatype)
    
#for dtype in [np.bool,np.uint8,np.int,np.int64,np.float16,np.float]:
#    show_dtype([0,-2,0.999999999999,21113,4,888888888,9e8],dtype)

##创建特定数据
###3行4列的0矩阵
#array_zeros = np.zeros((3,4))
#sprint(array_zeros)
###3行4列的1矩阵
#array_ones = np.ones((3,4))
#sprint(array_ones)
###3行4列的空矩阵，数值随机初始化
#array_empty = np.empty((3,4))
#sprint(array_empty)
###连续矩阵，给定范围和步长
#array_arange = np.arange(10,20,2)
#sprint(array_arange)
###连续矩阵，给定范围和步长,规定形状
#array_arange_reshape = np.arange(10,20,1).reshape((2,5))
#sprint(array_arange_reshape)
###线段型数据，给定范围和点数
#array_linspace = np.linspace(1,10,20)
#sprint(array_linspace)
###线段型数据，给定范围和点数,规定形状
#array_linspace_reshape = np.linspace(1,10,20).reshape((4,5))
#sprint(array_linspace_reshape)
    
#基础运算
#a = np.arange(1,8,2)
#b = np.array([10,25,-5,10.2])

#sprint(a)
#sprint(b)
#sprint(a-b)
#sprint(a*b)
#sprint(a**2)
#sprint(10*np.sin(a))
#sprint(a<3)
#sprint(a.dot(b))
#sprint(np.dot(a,b))


#a = np.arange(1,5).reshape(2,2)
#sprint(a)
#b = np.array([[0,1],[1,0]])
#sprint(b)
###矩阵没有交换律，左乘行变换，右乘列变换
#sprint(a.dot(b))
#sprint(b.dot(a))

###求矩阵的一些数字特征
#a = np.random.random((2,4))
#sprint(a)
##全空间
#sprint(np.sum(a))
#sprint(np.min(a))
#sprint(np.max(a))
#sprint(np.mean(a))
#sprint(np.median(a))
##列维度的数字特征，axis=0:跨列运算
#sprint(np.sum(a,axis=0))
#sprint(np.min(a,axis=0))
#sprint(np.max(a,axis=0))
#sprint(np.mean(a,axis=0))
#sprint(np.median(a,axis=0))
##行维度的数字特征，axis=1:跨行运算
#sprint(np.sum(a,axis=1))
#sprint(np.min(a,axis=1))
#sprint(np.max(a,axis=1))
#sprint(np.mean(a,axis=1))
#sprint(np.median(a,axis=1))

#A = np.arange(2,14).reshape((3,4))
#sprint(A)
##arg表示解的位置,抽成一维
#sprint(np.argmin(A))
#sprint(np.argmax(A))
##累加
#sprint(np.cumsum(A))
##对差
#sprint(np.diff(A))

#找出非0元素的行列索引
#A = np.identity(4)
#sprint(np.nonzero(A))
#for i in range(len(np.nonzero(A)[0])):
#    print('({},{})'.format(np.nonzero(A)[0][i],np.nonzero(A)[1][i]))

#排序
#A = np.arange(14,2,-1).reshape((3,4))
##多维默认axis=0
#np.random.shuffle(A)
#sprint(A)
##默认行排序
#sprint(np.sort(A))
##列排序 指定axis=0
#sprint(np.sort(A,axis=0))
    
#转置
#A=np.identity(4)
#np.random.shuffle(A)
#sprint(A)
#sprint(A.T)
#单位正交，A-1=AT
#sprint(A.T.dot(A))

#线性代数模块
#A = np.random.randint(10,size=(3,3))#随机产生0-9填充的3*3矩阵
#sprint(A)
##行列式
#sprint(np.linalg.det(A))
##逆
#sprint(np.linalg.inv(A))
##特征值和特征向量
#sprint(np.linalg.eig(A))
#用于对称矩阵的特征值特征向量
#A = np.random.randint(10,size=(3,3))
#A = A + A.T
#sprint(A)
#sprint(np.linalg.eig(A))
#sprint(np.linalg.eigh(A))
#仅特征值
#sprint(np.linalg.eigvals(A))
#sprint(np.linalg.eigvalsh(A))
##求解线性方程组Ax=b
#b=np.array([0,1,2])
#sprint(np.linalg.solve(A,b))
##奇异矩阵，会报错
#A = np.arange(0,9).reshape((3,3))#奇异矩阵
#sprint(np.linalg.solve(A,b))

#clip限制矩阵的最大值和最小值
#A = np.arange(0,9).reshape((3,3))
#sprint(A)
#sprint(np.clip(A,3,5))
















