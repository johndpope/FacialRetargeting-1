# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 17:07:33 2018

@author: Administrator
"""
from keras.layers import Reshape, LeakyReLU, Flatten, Activation, Input, Concatenate, Add
from keras.layers import Dense, Lambda, BatchNormalization
#from keras.utils import multi_gpu_model
from keras.models import Model
import keras.backend as K
from keras.engine.topology import Layer


class BiasChanneWise(Layer):
    def __init__(self, **kwargs): 
        super(BiasChanneWise, self).__init__(**kwargs)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer. 
        self.bias = self.add_weight(name='bias',
                            shape=(input_shape[1:]),
                            initializer='zeros',
                            trainable=True)
        super(BiasChanneWise, self).build(input_shape) # Be sure to call this at the end 
    def call(self, x): 
        return x+self.bias
    def compute_output_shape(self, input_shape): 
        return input_shape
    
class Bias(Layer):
    def __init__(self, **kwargs): 
        super(Bias, self).__init__(**kwargs)
    def build(self, input_shape):
        # Create a trainable weight variable for this layer. 
        self.bias = self.add_weight(name='bias',
                            shape=(input_shape[1],1),
                            initializer='zeros',
                            trainable=True)
        super(Bias, self).build(input_shape) # Be sure to call this at the end 
    def call(self, x): 
        return x+self.bias
    def compute_output_shape(self, input_shape): 
        return input_shape
    
def sampling(args):
    """Implements reparameterization trick by sampling
    from a gaussian with zero mean and std=1.
    Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    Returns:
        sampled latent vector (tensor)
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def get_gcn_vae_exp(T_k, support = 3, batch_size = 1, v = 11510, feature_dim = 9, input_dim = 11510 * 9, output_dim = 9499*9, vis = False, hidden_dim = 300,latent_dim = 25):
    g = [K.variable(_) for _ in T_k]
    def gcn(x):
        supports = list()
        for i in range(support):
            num , v , feature_shape = K.int_shape(x)
            LF = list()
            for j in range(batch_size):
                LF.append(K.expand_dims(K.dot(g[i], K.squeeze(K.slice(x, [j,0,0],[1,v,feature_shape]), axis = 0)), axis=0))
            supports.append(K.concatenate(LF,axis=0)) 
        x = K.concatenate(supports)
        return x
    def GConv(x, input_dim, output_dim, support, active_func = LeakyReLU(alpha=0.1)):#Activation('tanh')):#
        x = Lambda(gcn, output_shape=(v, support*input_dim))(x)
        x = Dense(output_dim)(x)
        x = active_func(x)
        return x
    inputs = Input(shape=(feature_dim * v, ))
    x = Reshape((v, feature_dim))(inputs)
    # One GCN Layer
    x = GConv(x, feature_dim, 128, support)
    x = GConv(x, 128, 64, support)
    x = GConv(x, 64, 64, support)
    x = GConv(x, 64, 64, support)
    x = GConv(x, 64, 1, support, Activation('tanh'))
    x = Flatten()(x)

    x = Dense(hidden_dim)(x)
    x = LeakyReLU(alpha=0.1)(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    x = Activation('sigmoid')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    encoder = Model(inputs,[z_mean, z_log_var, z], name = 'exp_encoder')
    code = Input(shape=(latent_dim, ))
    x = Dense(hidden_dim)(code)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(output_dim)(x)
    output = Activation('tanh')(x)

    decoder = Model(code, output)
    output = decoder(encoder(inputs)[2])
    gcn_vae = Model(inputs, output)
    if vis:
        gcn_vae.summary()
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.mean(K.abs(K.sum(kl_loss, axis=-1)))
    return kl_loss, encoder, decoder, gcn_vae

def get_gcn_encoder(T_k, support = 3, batch_size = 1, v = 11510, feature_dim = 9, input_dim = 11510 * 9, vis = False, hidden_dim = 300,latent_dim = 25):
    g = [K.variable(_) for _ in T_k]
    def gcn(x):
        supports = list()
        for i in range(support):
            num , v , feature_shape = K.int_shape(x)
            LF = list()
            for j in range(batch_size):
                LF.append(K.expand_dims(K.dot(g[i], K.squeeze(K.slice(x, [j,0,0],[1,v,feature_shape]), axis = 0)), axis=0))
            supports.append(K.concatenate(LF,axis=0)) 
        x = K.concatenate(supports)
        return x
    def GConv(x, input_dim, output_dim, support, active_func = LeakyReLU(alpha=0.1)):#Activation('tanh')):#
        x = Lambda(gcn, output_shape=(v, support*input_dim))(x)
        x = Dense(output_dim)(x)
        x = active_func(x)
        return x
    inputs = Input(shape=(feature_dim * v, ))
    x = Reshape((v, feature_dim))(inputs)
    # One GCN Layer
    x = GConv(x, feature_dim, 128, support)
    x = GConv(x, 128, 64, support)
    x = GConv(x, 64, 64, support)
    x = GConv(x, 64, 64, support)
    x = GConv(x, 64, 1, support, Activation('tanh'))
    x = Flatten()(x)

    x = Dense(hidden_dim)(x)
    x = LeakyReLU(alpha=0.1)(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    x = Activation('sigmoid')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    encoder = Model(inputs,[z_mean, z_log_var, z], name = 'exp_encoder')
    return encoder


def get_gcn_decoder(T_k, support = 3, batch_size = 1, v = 9499, feature_dim = 9, output_dim = 9499 * 9, vis = False, hidden_dim = 300,latent_dim = 25):
    g = [K.variable(_) for _ in T_k]
    def gcn(x):
        supports = list()
        for i in range(support):
            num , v , feature_shape = K.int_shape(x)
            LF = list()
            for j in range(batch_size):
                LF.append(K.expand_dims(K.dot(g[i], K.squeeze(K.slice(x, [j,0,0],[1,v,feature_shape]), axis = 0)), axis=0))
            supports.append(K.concatenate(LF,axis=0)) 
        x = K.concatenate(supports)
        return x
    def GConv(x, input_dim, output_dim, support, active_func = LeakyReLU(alpha=0.1)):#Activation('tanh')):#
        x = Lambda(gcn, output_shape=(v, support*input_dim))(x)
        x = Dense(output_dim)(x)
        x = Bias()(x)
        x = active_func(x)
        return x
    code = Input(shape=(latent_dim, ))
    x = Dense(hidden_dim)(code)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(v)(x)
    x = Reshape((v, 1))(x)
    x = GConv(x, 1, 64, support)
    x = GConv(x, 64, 64, support)
    x = GConv(x, 64, 64, support)
    x = GConv(x, 64, 128, support)    
    x = GConv(x, 128, feature_dim, support, Activation('tanh'))
    output = Flatten()(x)
    decoder = Model(code, output)
    return decoder

def get_transfer(input_dim, output_dim, hidden_dim = 100):
    code_in = Input(shape=(input_dim, ))
    x1 = Dense(hidden_dim)(code_in)
    x1 = LeakyReLU(alpha=0.1)(x1)
    x2 = Dense(hidden_dim)(x1)
    x2 = LeakyReLU(alpha=0.1)(x2)
    x2 = Concatenate()([x1,x2])
    x3 = Dense(hidden_dim)(x2)
    x3 = LeakyReLU(alpha=0.1)(x3)
    x3 = Concatenate()([x1,x2,x3])
    x4 = Dense(output_dim)(x3)
    x4 = LeakyReLU(alpha=0.1)(x4)
    transfer = Model(code_in, x4)
    return transfer
    
if __name__ == '__main__':
    """
    Testing code
    """
    #generate_mesh("Normalized_feature/4.txt", "generate_1.obj")
    start = True
