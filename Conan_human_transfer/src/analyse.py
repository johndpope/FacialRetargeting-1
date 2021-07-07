#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:19:17 2018

@author: jzh
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.manifold
import sklearn.decomposition
from data_utils import draw, deduce_mean
plt.switch_backend('agg')


# number of nodes to use
numnodes = 210
t = 5
n = 20
tup = (None,150)#47,2*47,3*47,4*47,5*47,6*47,7*47,8*47,9*47)
encode_data_exp = np.load('/raid/jzh/FeatureDistangle/data/encode_data/find_mean.npy')[:150]
#encode_data_mean = np.load('../data/encode_data/people_mean_.npy')[:5*47]
#encode_data_res = np.load('../data/encode_data/res31.npy')
#encode_data_mean = np.load('../data/Feature.npy')[:5*47]
#encode_data_mean = deduce_mean(tup,encode_data_mean)
#encode_data_exp = deduce_mean(tup,encode_data_exp)
#encode_data = encode_data_var
#embed =sklearn.decomposition.PCA(n_components=2).fit_transform(encode_data)
#embed_1 =sklearn.manifold.TSNE(n_components=2).fit_transform(encode_data_1)
embed =sklearn.manifold.TSNE(n_components=2).fit_transform(encode_data_exp)

color = ['ro','bo','co','ko','mo', 'r*','m*','b.','bo','k.','k*','y*','ro','bv','rv','kv','yv','mv','cv']

draw(color, tup, embed)

