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