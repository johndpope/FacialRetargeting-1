# -*- coding: utf-8 -*- 
"""
Created on Mon Jul 23 17:07:33 2018

@author: Administrator
"""
import numpy as np, keras.backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from src.VAE import get_gcn_vae_exp, get_gcn_decoder, get_gcn_encoder, get_transfer
from src.data_utils import normalize_fromfile, denormalize_fromfile, data_recover, batch_change
#from src.get_mesh import get_mesh
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
#from src.mesh import V2M2
ref_name = 'data/people/Mean_Face.obj'
def get_general_laplacian(adj):
    return (sp.diags(np.power(np.array(adj.sum(1)), 1).flatten(), 0) - adj) * sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
    
    
    
def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = (sp.eye(adj.shape[0], dtype=np.float32)) - adj_normalized
    return laplacian


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = (eigsh(laplacian, 1, which='LM', return_eigenvectors=False))[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = 2.0 / largest_eigval * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print(('Calculating Chebyshev polynomials up to order {}...').format(k))
    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    T_k = [i.astype(np.float32) for i in T_k]
    return T_k

class moji(object):

    def __init__(self, input_dim, output_dim, prefix, suffix, lr, load, feature_dim=9, latent_dim=25, kl_weight=0.000005, batch_size=1, MAX_DEGREE=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prefix = prefix
        self.suffix = suffix
        self.load = load
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.v = int(input_dim / feature_dim)
        self.out_v = int(output_dim / feature_dim)
        self.hidden_dim = 100
        self.lr = lr
        self.kl_weight = kl_weight
        self.M_list = np.load(('data/{}/max_data.npy').format(self.prefix))
        self.m_list = np.load(('data/{}/min_data.npy').format(self.prefix))
        self.p_M_list = np.load(('data/{}/max_data.npy').format('people'))
        self.p_m_list = np.load(('data/{}/min_data.npy').format('people'))
        self.batch_size = batch_size
        self.build_model(MAX_DEGREE)


    def build_model(self, MAX_DEGREE):
        SYM_NORM = True
        A = sp.load_npz(('data/people/FWH_adj_matrix.npz'))
        L = normalized_laplacian(A, SYM_NORM)
        T_k = chebyshev_polynomial(rescale_laplacian(L), MAX_DEGREE)
        support = MAX_DEGREE + 1
        _, self.encoder, self.decoder, self.gcn_vae_exp = get_gcn_vae_exp(T_k, support, batch_size=self.batch_size, \
                                                                          feature_dim=self.feature_dim, v=self.v, \
                                                                          input_dim=self.input_dim, output_dim = self.output_dim)
        self.target_exp = Input(shape=(self.output_dim,))
        real = self.gcn_vae_exp.get_input_at(0)
        ratio = K.variable(self.M_list - self.m_list)
        s = K.variable(self.m_list + self.M_list)
        # L2 when xyz, L1 when rimd

        self.exp_loss = K.mean(K.abs((self.target_exp - self.gcn_vae_exp(real)) * ratio )) / 1.8 #+ self.away_loss
        

        weights = self.decoder.trainable_weights
        self.regular_loss = 0
        for w in weights:
            self.regular_loss += 0.000002 * K.sum(K.square(w))
        self.loss = self.exp_loss + self.regular_loss
        training_updates = (Adam(lr=self.lr)).get_updates(weights, [], self.loss)
        self.train_func = K.function([real, self.target_exp], [self.exp_loss, self.loss, self.regular_loss], training_updates)
        self.test_func = K.function([real, self.target_exp], [self.exp_loss, self.loss])
        if self.load:
            self.load_models()
        else:
            self.encoder.load_weights('model/encoder_exp_people.h5')

class triplemoji(moji):
    def build_model(self, MAX_DEGREE):

        SYM_NORM = True
        A = sp.load_npz(('data/people/FWH_adj_matrix.npz'))
        L = normalized_laplacian(A, SYM_NORM)
        T_k = chebyshev_polynomial(rescale_laplacian(L), MAX_DEGREE)
        support = MAX_DEGREE + 1
         #self.people_encoder = get_gcn_encoder(T_k, support, batch_size=self.batch_size, feature_dim=self.feature_dim, v=self.v, input_dim=self.input_dim)
        _,self.people_encoder, self.people_decoder,_ = get_gcn_vae_exp(T_k, support, batch_size=self.batch_size, \
                                                                        feature_dim=self.feature_dim, v=self.v,\
                                                                        input_dim=self.input_dim, output_dim = self.input_dim)
        A = sp.load_npz(('data/{}/{}_adj_matrix.npz').format(self.prefix, self.prefix))
        L = normalized_laplacian(A, SYM_NORM)
        T_k = chebyshev_polynomial(rescale_laplacian(L), MAX_DEGREE)
        _, self.dog_encoder, self.dog_decoder, self.dog_gcn_vae_exp = get_gcn_vae_exp(T_k, support, batch_size=self.batch_size, \
                                                                                      feature_dim=self.feature_dim, v=self.out_v, \
                                                                                      input_dim=self.output_dim, output_dim = self.output_dim)
        
        # People to dog
        self.PTD = get_transfer(25,25)
        
        # Inputs
        self.conan_exp = Input(shape=(self.output_dim,))
        self.conan_P = Input(shape=(self.output_dim,))
        self.conan_N = Input(shape=(self.output_dim,))
        
        real = self.people_encoder.get_input_at(0)
        
        # Outputs
        output = self.dog_decoder(self.PTD(self.people_encoder(real)[0]))
        #output = self.dog_decoder(self.dog_encoder(real)[0])
        
        # simplest way of our Model
        self.gcn_vae_exp = Model(real, output)

        z_mean, z_log_var, z = self.people_encoder(real)
        code_people = z_mean
        #code_people = z
        #var_people = z_log_var/K.sqrt(K.sum(K.square(z_log_var)))
        
        ratio = K.variable(self.M_list - self.m_list)
        s = K.variable(self.m_list + self.M_list)
        # L2 when xyz, L1 when rimd

        z_mean, z_log_var, z = self.dog_encoder(self.conan_exp)
        code_conan = z_mean
        #code_conan = z
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        self.kl_loss = K.mean(K.abs(K.sum(kl_loss, axis=-1)))

        z_mean, z_log_var, z = self.dog_encoder(self.conan_P)
        code_P = z_mean
        #code_P = z

        z_mean, z_log_var, z = self.dog_encoder(self.conan_N)
        code_N = z_mean
        #code_N = z

        data_conan = self.dog_decoder(code_conan)
        data_people = self.dog_decoder(self.PTD(code_people))
        
        data_P = self.dog_decoder(code_P)
        data_N = self.dog_decoder(code_N)

        d=0.0
        self.group_loss = K.mean(K.abs((data_conan - data_people)*ratio)) / 1.8


        self.positive_loss = K.mean(K.abs(code_P - self.PTD(code_people)))
        self.negative_loss = K.mean(K.abs(code_N - self.PTD(code_people)))

        m=0.2
        self.tpl_loss = K.relu(self.positive_loss - self.negative_loss + m)

        
        self.re_loss_P = K.mean(K.abs((self.conan_P - self.dog_gcn_vae_exp(self.conan_P)) * ratio )) / 1.8 
        self.re_loss_N = K.mean(K.abs((self.conan_N - self.dog_gcn_vae_exp(self.conan_N)) * ratio )) / 1.8
        self.re_loss = self.re_loss_P + self.re_loss_N
        
        
        weights = self.PTD.trainable_weights# + self.gcn_vae_exp.trainable_weights# + self.dog_gcn_vae_exp.trainable_weights
        
        self.regular_loss = 0
        for w in weights:
            self.regular_loss += 0.00001 * K.sum(K.square(w))
        
       
        self.loss = self.re_loss + self.tpl_loss + self.regular_loss    #easy_tpl_and_positive_loss
        #self.loss = self.re_loss + self.positive_loss * 5 + self.regular_loss                    #only_positive_loss 

        training_updates = (Adam(lr=self.lr)).get_updates(weights, [], self.loss)

        #self.train_func = K.function([real, self.conan_exp], [self.group_loss, self.loss, self.regular_loss], training_updates)
        #self.test_func = K.function([real, self.conan_exp], [self.group_loss, self.loss])
        self.train_func = K.function([real, self.conan_P, self.conan_N], [self.tpl_loss, self.positive_loss, self.re_loss, self.loss, self.regular_loss], training_updates)
        self.test_func = K.function([real, self.conan_P, self.conan_N], [self.tpl_loss, self.positive_loss, self.re_loss, self.loss])

        self.people_encoder.load_weights('model/encoder_exp_people.h5')
        self.people_decoder.load_weights('model/decoder_exp_people.h5')
        self.dog_encoder.load_weights('model/encoder_exp_conan.h5')
        self.dog_decoder.load_weights('model/decoder_exp_conan.h5')
        

        if self.load:
            self.load_models()
        

    def save_models(self):
        #self.gcn_vae_exp.save_weights(('model/gcn_vae_exp{}{}.h5').format(self.prefix, self.suffix))
        self.PTD.save_weights(('model/PTD_{}{}.h5').format(self.prefix, self.suffix))
        
    def load_models(self):
        #self.gcn_vae_exp.load_weights(('model/gcn_vae_exp{}{}.h5').format(self.prefix, self.suffix))
        self.people_encoder.load_weights('model/encoder_exp_people.h5')
        self.people_decoder.load_weights('model/decoder_exp_people.h5')
        self.dog_encoder.load_weights('model/encoder_exp_conan.h5')
        self.dog_decoder.load_weights('model/decoder_exp_conan.h5')
        self.PTD.load_weights(('model/PTD_{}_tpl.h5').format(self.prefix))
        
        
    def test(self, limit=5, filename='test', people_id=142):
        data = np.load('data/people/test_data.npy')
        data_array = data.copy()
        data_test=data.copy()
        normalize_fromfile(data_array, self.p_M_list, self.p_m_list)
        feature_exp = denormalize_fromfile(self.dog_decoder.predict(self.PTD.predict(self.people_encoder.predict(data_array, batch_size=self.batch_size)[0],batch_size=self.batch_size)), self.M_list, self.m_list)
        people_exp = denormalize_fromfile(self.people_decoder.predict(self.people_encoder.predict(data_array, batch_size=self.batch_size)[2],batch_size=self.batch_size), self.p_M_list, self.p_m_list)
        import shutil, os
        from src.measurement import write_align_mesh
        shutil.rmtree('data/mesh')
        os.mkdir('data/mesh')
        ref_exp = 'data/{}/{}_neutral.obj'.format(self.prefix,self.prefix)
        for i in range(data_array.shape[0]):
            V2M2(get_mesh(ref_exp, data_recover(feature_exp[i])), ('data/mesh/exp_{}_{}.obj').format(self.prefix, i), v_num = int(self.out_v), ref_name = ref_exp)
            
            write_align_mesh(('data/mesh/exp_{}_{}.obj').format(self.prefix, i),'data/{}/{}_neutral.obj'.format(self.prefix, self.prefix),('data/mesh/aligned_exp_{}_{}.obj').format(self.prefix, i))
            V2M2(get_mesh('data/people/Mean_Face.obj', data_recover(people_exp[i])), ('data/mesh/people_exp_{}_{}.obj').format(self.prefix, i), v_num = int(self.v), ref_name = 'data/people/Mean_Face.obj')    
if __name__ == '__main__':
    start = True