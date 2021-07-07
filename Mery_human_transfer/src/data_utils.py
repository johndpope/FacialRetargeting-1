# -*- coding: utf-8 -*-
"""
Created on Sat Sept 24 10:12:01 2018

@author: Administrator
"""

import os
import numpy as np
import matplotlib.pyplot as plt
#change_length = 9499
#change_length = 12942
change_length = 1895
unit = 9
delta = np.array([1,0,0,1,0,1,0,0,0])
cross_id = np.tile(delta, change_length)
def batch_change(data):
    '''
    to change the data by divide it to some 'unit'
    and reduce 'delta' in each unit
    the whole change step will take 'change_length' steps
    meaning only use data[: change_length * unit] 
    '''
    dim = data.shape[0]
    length = int(dim/9)
    cross = np.tile(delta, length)
    return data - cross

def data_interpolation(data_array, alpha = 0.5):
    '''
    interpolate data
    'alpha' is the constant of interpolation
    '''
    num,dim = data_array.shape
    whole_data = np.zeros((int(num*(num-1)/2 + num), dim))
    k = 0
    print((num,dim,int(num*(num-1)/2)))
    for i in range(num):
        print(i)
        for j in range(i,num):
            whole_data[k] = alpha * data_array[i] + (1-alpha) * data_array[j]
            k = k+1
    return whole_data

def load_data_fromfile(filename, M_list, m_list, **kwargs):
     data = batch_change(np.fromfile(filename),change_length,unit,delta)
     if kwargs=={}:
         data = normalize_fromfile(data[np.newaxis,:], M_list, m_list)
     else:
         data = normalize_fromfile(data[kwargs['filter_data']][np.newaxis,:], M_list, m_list)
     
     return data



def data_recover(data):
    '''
    recover data from minused data
    '''
    dim = data.shape[0]
    length = int(dim/9)
    cross = np.tile(delta, length)
    return data + cross
    

def normalize(data, a = 0.9 ,epsilon = 10e-6):
    num,dim = data.shape
    for i in range(dim):
        M = np.max(data[:,i])
        m = np.min(data[:,i])
        if M-m < epsilon:
            M = M+epsilon
            m = m-epsilon
            
        data[:,i] = 2 * a * (data[:,i]-m) / (M-m) -a
    return data

def save_normalize_list(array, data, suffix = 'mery', a = 0.9 ,epsilon = 10e-6):
    num,dim = data.shape
    M_list = np.zeros_like(array)
    m_list = np.zeros_like(array)
    for i in range(dim):
        M = np.max(data[:,i])
        m = np.min(data[:,i])
        if M-m < epsilon:
            M = M+epsilon
            m = m-epsilon
        M_list[i] = M
        m_list[i] = m
    np.save('Max_list_{}'.format(suffix),M_list)
    np.save('Min_list_{}'.format(suffix),m_list)
    return array


def denormalize_fromfile(array, M_list, m_list, a = 0.9):
    num,dim = array.shape
    for i in range(dim):
        M = M_list[i]
        m = m_list[i]
        array[:, i] = (array[:,i]+a)*(M-m)/(2*a)+m
    return array



def normalize_fromfile(array, M_list, m_list, a = 0.9):
    num,dim = array.shape
    for i in range(dim):
        M = M_list[i]
        m = m_list[i]
        array[:, i] = 2 * a * (array[:,i]-m) / (M-m) -a
    return array

def deduce_mean(tup, data_array):
    data = data_array.copy()
    for i in range(len(tup)-1):
        data[tup[i]:tup[i+1]] -= np.mean(data[tup[i]:tup[i+1]], axis = 0)
        
    return data
def draw(color, tup, embed):
    n = min(len(color), len(tup)-1)
    plt.savefig('pict.png')
    for i in range(n):
        plt.plot(embed[tup[i]:tup[i+1],0],embed[tup[i]:tup[i+1],1],color[i])
    
    plt.savefig('draw_embed')
    #plt.show()cd 

def reduce_normalize_list(dir_name):
    epsilon = 10e-6
    max_data = np.fromfile(os.path.join(dir_name,'max_data.dat')) - cross_id
    min_data = np.fromfile(os.path.join(dir_name,'min_data.dat')) - cross_id
    for i in range(min_data.shape[0]):
        if abs(min_data[i] - max_data[i]) < epsilon:
            min_data[i] -= epsilon
            max_data[i] += epsilon
    np.save(os.path.join(dir_name,'max_data'), max_data)        
    np.save(os.path.join(dir_name,'min_data'), min_data)  

def concate_data():
#    whole_data = np.vstack((np.fromfile('/raid/jzh/2Moji/data/dog/Avatar_47/RimdFeature1123/{}_new.dat'.format(i)) - cross_id for i in range(1,48)))
#    np.save('/raid/jzh/2Moji/data/dog/exp',whole_data)
#    whole_data = np.vstack((np.fromfile('/raid/jzh/COMA_data/RimdFeature1030/MeanFace/COMA_mean_{}.dat'.format(i)) - cross_id for i in range(13)))
#    np.save('/raid/jzh/COMA_data/RimdFeature1030/MeanFace',whole_data)
#    for people_id in range(1,151):
#        whole_data = np.vstack((np.fromfile('/raid/jzh/RimdFeature1218/Tester_{}/pose_{}.dat'.format(people_id, i)) - cross_id for i in range(20)))
#        np.save('/raid/jzh/RimdFeature1218/gather_feature/Featur{}'.format(people_id),whole_data)
    train_data = np.vstack((np.load('/raid/jzh/Mery_moji/Mery_moji_5000/data/mery/interpolated_data/shape_{}_head.npy'.format(i)) - cross_id for i in range(0,5000)))
    #test_data = np.vstack((np.load('/raid/jzh/Mery_moji/Head_moji/data/mery/Mery_head/data/shape_{}_head.npy'.format(i)) for i in range(400,500)))
    np.save('/raid/jzh/Mery_moji/Mery_moji_5000/data/mery/exp.npy',train_data)
    #np.save('/raid/jzh/Mery_moji/Head_moji/data/mery/test.npy',test_data)
def compare():
    data_array = np.load('/raid/jzh/FeatureDistangle/data/coma/train_data/Feature1.npy')
    mean_exp = np.load('/raid/jzh/FeatureDistangle/data/coma/MeanFace.npy')
    neutral_face = np.repeat(data_array[:1],13,axis = 0)   
    
    return data_array - (mean_exp + neutral_face)
    # return mean_exp[1]-mean_exp[0]
    
if __name__ == '__main__':
    start = True

    concate_data()
    #data = np.load('/raid/jzh/Mery_moji/Mery_moji_5000/data/mery/exp.npy')
    #save_normalize_list(data[0], data)





    
