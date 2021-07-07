import pyigl as igl
import numpy as np
import scipy
import scipy.sparse as sp

def e2p(m):
    if isinstance(m, igl.eigen.MatrixXd):
        return np.array(m, dtype='float64', order='C')
    elif isinstance(m, igl.eigen.MatrixXi):
        return np.array(m, dtype='int32', order='C')
    elif isinstance(m, igl.eigen.MatrixXb):
        return np.array(m, dtype='bool', order='C')
    elif isinstance(m, igl.eigen.SparseMatrixd):
        coo = np.array(m.toCOO())
        I = coo[:, 0]
        J = coo[:, 1]
        V = coo[:, 2]
        return sp.coo_matrix((V,(I,J)), shape=(m.rows(),m.cols()), dtype='float64')
    elif isinstance(m, igl.eigen.SparseMatrixi):
        coo = np.array(m.toCOO())
        I = coo[:, 0]
        J = coo[:, 1]
        V = coo[:, 2]
        return sp.coo_matrix((V,(I,J)), shape=(m.rows(),m.cols()), dtype='int32')

V = igl.eigen.MatrixXd()
F = igl.eigen.MatrixXi()
#name = 'dog'
#igl.readOBJ('/raid/jzh/2Moji/data/{}/{}_netural.obj'.format(name,name), V, F)
igl.readOBJ('/raid/jzh/Mery_moji/Head_moji/Mery_head/neutral_head.obj', V, F)

L = igl.eigen.SparseMatrixd()
igl.cotmatrix(V, F, L)
A = e2p(L)
print(A)
c = A.col
r = A.row
adj = sp.coo_matrix((np.ones(c.shape[0]), (c, r)),
                    shape=(A.shape[0], A.shape[0]), dtype=np.float32).tocsr()-sp.eye(A.shape[0])
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
sp.save_npz('meryhead_adj_matrix',adj)
temp = sp.load_npz('meryhead_adj_matrix.npz')
print(temp)
