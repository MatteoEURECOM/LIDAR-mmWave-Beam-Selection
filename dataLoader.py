import pandas as pd
import numpy as np
import scipy.io


'''S008 mean and standard deviation to normalize GPS inputs'''
s008_mean=[757.8,555.4,2.1]
s008_SD=[4.5,68.0,0.9]

def lidar_to_2d(lidar_data):
    '''
    :param lidar_data: Batch of 3D LIDAR images
    :return: Batch of 2D LIDAR images converted by flattening the Z-axis using the max value.
    '''
    lidar_data1 = np.zeros_like(lidar_data)[:, :, :, 1]
    lidar_data1[np.max(lidar_data == 1, axis=-1)] = 1
    lidar_data1[np.max(lidar_data == -2, axis=-1)] = -2
    lidar_data1[np.max(lidar_data == -1, axis=-1)] = -1
    return lidar_data1

def lidar_to_2d_summing(lidar_data):
    '''
    :param lidar_data: Batch of 3D LIDAR images
    :return: Batch of 2D LIDAR images converted by flattening the Z-axis using the mean value.
    '''
    lidar_data1 = np.sum(lidar_data,axis=3)/10.0
    tx_x,tx_y,tx_z=np.where(lidar_data1 == -1)
    rx_x,rx_y,rx_z=np.where(lidar_data1 == -2)
    lidar_data1[tx_x,tx_y,tx_z]=-1
    lidar_data1[rx_x,rx_y,rx_z]=-2
    return lidar_data1

def load_dataset(filename,FLATTENED,SUM):
    '''
    :param filename of the dataset to load and LIDAR format parameteres
    :return: [GPS,LIDAR,labels,LOS/NLOS] dataset with LOS available only for s008 and s009
    '''
    npzfile = np.load(filename)
    POS=npzfile['POS']
    for i in range(0,3):
        POS[:,i]=(POS[:,i]-s008_mean[i])/s008_SD[i]
    LIDAR=npzfile['LIDAR']
    if(FLATTENED and SUM):
        LIDAR = np.expand_dims(lidar_to_2d_summing(LIDAR), axis=3)
    elif(FLATTENED and not SUM):
        LIDAR = np.expand_dims(lidar_to_2d(LIDAR), axis=3)
    Y=npzfile['Y']
    LOS=npzfile['LOS']
    return POS,((LIDAR+2.0)/3.0).astype(np.float32),Y,LOS

