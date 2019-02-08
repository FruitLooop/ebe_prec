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
        self.gl_identity = np.identity(self.global_n)

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

    def get_inv_dia(self):
        return self.diag_inv_sqrt(self.global_diag)

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

    def get_element_diag(self, size, cell_id):
        n_c = size
        ce_matrix = np.zeros(shape=[n_c, n_c])
        gl_diag_scaler_inv = self.get_inv_dia()

        for i in range(ce_matrix.shape[1]):
            if cell_id[i] != -1 and cell_id[i] < gl_diag_scaler_inv.shape[1]:
                        value1 = cell_id[i]
                        ce_matrix[i, i] = gl_diag_scaler_inv[value1, value1]
        return ce_matrix

    def precon(self, data, diag):

        for i in range(len(data)):
            kc = np.asmatrix(data[i][0])
            data_pos = data[i][1]
            dc = np.asmatrix(np.array(diag[i]))

            diag_inv = self.get_element_diag(kc.shape[0], data_pos)
            term1 = kc - dc
            term2 = np.matmul(diag_inv, np.matmul(term1, diag_inv))
            identity_cell = np.identity(kc.shape[0])

            cell_P = identity_cell + term2
            cell_chol = np.linalg.cholesky(cell_P)

            self.global_C = self.assembly(self.global_C, cell_chol, data_pos)

        return self.gl_identity + self.global_C


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
weight = 1e-2
C = ebe_prec(global_n, weight)

data_diag = C.element_diag_scaler(data)
chol = C.precon(data, data_diag)
chol_T = chol.T

P_L = inv(chol)
P_R = inv(chol_T)

prec_Mat = np.matmul(np.matmul(P_L, global_Mat), P_R)

print(LA.cond(global_Mat))
print(LA.cond(prec_Mat))

