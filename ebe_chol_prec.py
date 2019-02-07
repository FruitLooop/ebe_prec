import numpy as np
from numpy import linalg as LA
import scipy.io
import os
from numpy.linalg import inv

class ebe_prec:

    def __init__(self, global_n, weight):
        self.global_n = global_n
        self.global_diag = np.zeros(shape=[self.global_n, self.global_n])
        self.global_C = np.zeros(shape=[self.global_n, self.global_n])
        self.data = []
        self.weight = weight

    def element_diag_scaler(self, data):
        data_diag = []

        for i in range(len(data)):
            cell_K = data[i][0]
            cell_number = cell_K.shape[0]
            cell_pos = data[i][1]
            cell_diag = np.zeros(shape=[cell_number, cell_number])

            for j in range(cell_number - 1):
                if np.count_nonzero(cell_K) == 0:
                    cell_diag[j, j] = 0
                else:
                    row_sum = np.sum(np.absolute(cell_K[j]))
                    cell_diag[j, j] = row_sum

            cell_diag = self.weight * cell_diag
            self.assembly(self.global_diag, cell_diag, cell_pos)
            data_diag.append([cell_diag])

        return data_diag

    def diag_inv_sqrt(self, global_diag):
        global_inv_diag = np.zeros(shape=[self.global_n, self.global_n])
        for i in range(global_diag.shape[1]):
            if global_diag[i, i] != 0:
                global_inv_diag[i, i] = 1/np.sqrt(global_diag[i, i])
        return global_inv_diag

    def assembly(self, gl_matrix, ce_matrix, cell_id):
        for j in range(ce_matrix.shape[1]):
            if cell_id[j] < gl_matrix.shape[1]:
                for k in range(ce_matrix.shape[1]):
                    if cell_id[j] != -1 and cell_id[k] != -1 and cell_id[k] < gl_matrix.shape[1]:
                        value1 = cell_id[j]
                        value2 = cell_id[k]
                        gl_matrix[value1, value2] += ce_matrix[j, k]
        return gl_matrix

    def precon(self, data, diag):
        gl_diag_inv = self.diag_inv_sqrt(self.global_diag)
        identity_global = np.identity(self.global_n)

        for i in range(len(data)):
            data_k = np.asmatrix(data[i][0])
            data_pos = data[i][1]
            data_dia = np.asmatrix(np.array(diag[i]))

            identity_cell = np.identity(data_k.shape[0])
            boolean_kc = np.zeros(shape=[self.global_n, self.global_n])
            boolean_dc = np.zeros(shape=[self.global_n, self.global_n])
            boolean_identity = np.zeros(shape=[self.global_n, self.global_n])

            kc = self.assembly(boolean_kc, data_k, data_pos)
            dc = self.assembly(boolean_dc, data_dia, data_pos)
            term1 = kc - dc
            term2 = np.matmul(gl_diag_inv, np.matmul(term1, gl_diag_inv))
            cell_P = identity_global + term2

            identity_cell_global = self.assembly(boolean_identity, identity_cell, data_pos)
            cell_chol = np.linalg.cholesky(cell_P)
            cell_chol_diff = cell_chol - identity_cell_global

            self.global_C = self.global_C + cell_chol_diff

        return identity_global + self.global_C



path = 'q4/mesh11x11.gid/mfsc/'
data = []
for file1 in os.listdir(path):
    file1name = os.fsdecode(file1)
    if file1.endswith('.mm') and file1.startswith('ke_0_1'):
        file2 = file1name.replace('ke', 'eq').replace('.mm', '.txt')
        file2name = os.fsdecode(file2)

        k_cell = np.matrix(scipy.io.mmread(os.path.join(path, file1name)).toarray())
        position = np.loadtxt(os.path.join(path, file2name))
        position = list(map(int, position))
        data.append([k_cell, position])
        continue

global_Mat = np.matrix(scipy.io.mmread(path + 'A_0.1.mm').toarray())
global_n = global_Mat.shape[1]
weight = 1e-4
C = ebe_prec(global_n, weight)

data_diag = C.element_diag_scaler(data)
chol = C.precon(data, data_diag)
chol_T = chol.T

P_L = inv(chol)
P_R = inv(chol_T)

prec_Mat = np.matmul(np.matmul(P_L, global_Mat), P_R)

print(LA.cond(global_Mat))
print(LA.cond(prec_Mat))

