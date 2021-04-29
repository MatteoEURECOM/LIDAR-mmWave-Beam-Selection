from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Conv2D,PReLU, ReLU, Softmax, add, \
    Flatten, MaxPooling2D, Dense, Reshape, Input, Dropout, concatenate, GaussianNoise
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.utils
from keras.regularizers import l2,l1
from tensorflow.keras import initializers
from non_local import non_local_block


def MIXTURE():
    input_lid = Input(shape=(20, 200, 1))
    layer = Conv2D(5, kernel_size=(5, 5), activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(input_lid)
    layer = Conv2D(5, kernel_size=(5, 5), activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    layer = Conv2D(5, kernel_size=(5, 5), strides=2, activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    layer = Conv2D(5, kernel_size=(5, 5), activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    layer = Conv2D(5, kernel_size=(5, 5), strides=2, activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    layer = Conv2D(1, kernel_size=(3, 3), strides=(1, 2), activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    layer = Flatten()(layer)
    out_lid = Dense(16, activation='relu')(layer)
    '''GPS branch'''
    input_coord = Input(shape=(2))
    '''Concatenation'''
    concatenated = concatenate([out_lid, input_coord])
    layer = Dense(64, activation='relu')(concatenated)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(64, activation='relu')(layer)
    predictions = Dense(256, activation='softmax')(layer)
    architecture = Model(inputs=[input_lid, input_coord], outputs=predictions)
    return architecture

def NON_LOCAL_MIXTURE():
    '''NONLOCAL MODEL'''
    '''LIDAR branch'''
    input_lid = Input(shape=(20, 200, 1))
    layer = Conv2D(5, kernel_size=(5, 5), activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(input_lid)
    layer = Conv2D(5, kernel_size=(5, 5), activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    layer = Conv2D(5, kernel_size=(5, 5), strides=2, activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    layer = Conv2D(5, kernel_size=(5, 5), activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    layer = Conv2D(5, kernel_size=(5, 5), strides=2, activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(layer)
    NLA= non_local_block(layer,intermediate_dim=2,mode='embedded')
    layer = Conv2D(1, kernel_size=(3, 3), strides=(1, 2), activation='relu', padding="SAME", kernel_initializer=initializers.HeUniform)(NLA)
    layer = Flatten()(layer)
    out_lid = Dense(16, activation='relu')(layer)
    '''GPS branch'''
    input_coord = Input(shape=(2))
    '''Concatenation'''
    concatenated = concatenate([out_lid, input_coord])
    layer = Dense(64, activation='relu')(concatenated)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(64, activation='relu')(layer)
    predictions = Dense(256, activation='softmax')(layer)
    architecture = Model(inputs=[input_lid, input_coord], outputs=predictions)
    return architecture

