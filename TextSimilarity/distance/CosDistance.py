#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import dot,array
import numpy as np
from gensim import  matutils

def similarity(array1, array2):

    return dot(matutils.unitvec(array1), matutils.unitvec(array2))


def n_similarity(ds1, ds2):

    v1 = [vec for vec in ds1]
    v2 = [vec for vec in ds2]
    return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))

if __name__ == '__main__':
    arr_1 = [1.0, 0.0, 0.0, 0.0, 1.0]
    arr_2 = [1.0, 1.0, 0.0, 0.0, 1.0]
    arr_1=np.array(arr_1)
    arr_2 = np.array(arr_2)
    print(similarity(array1=arr_1,array2=arr_2))
    list_1=[[1.0, 0.0, 0.0, 0.0, 1.0],[1.0, 0.0, 0.0, 0.0, 1.0]]
    list_2=[[1.0, 1.0, 0.0, 0.0, 1.0]]
    print(n_similarity(list_1,list_2))