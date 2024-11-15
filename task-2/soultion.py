import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pyqiopt as pq

def A1_calc(m, n, k, C):
    I_n = np.ones(n, dtype=int)
    I_m = np.ones(m, dtype=int)
    I_m = I_m.reshape((1, m))
    I_n = I_n.reshape((1, n))

    E_n = np.eye(n)
    E_m = np.eye(m)
    
    matrix = np.zeros((k, k))

    np.fill_diagonal(matrix[1:], 1)

    D = matrix.T
    
    return np.kron(np.kron(E_n, C), D)

def A2_calc(m, n, k):
    I_n = np.ones(n, dtype=int)
    I_m = np.ones(m, dtype=int)
    I_m = I_m.reshape((1, m))
    I_n = I_n.reshape((1, n))

    E_k = np.eye(k)
    E_m = np.eye(m)
    
    E_m[0,0] = 0
    
    return np.kron(np.kron(I_n, E_m), E_k), np.ones((k * m, 1))

def A3_calc(m, n, k, B):
    I_k = np.ones(k, dtype=int)
    I_m = np.ones(m, dtype=int)
    I_m = I_m.reshape((1, m))
    I_k = I_k.reshape((1, k))

    E_n = np.eye(n)
    E_m = np.eye(m)
    
    return np.kron(np.kron(E_n, B), I_k)/10, np.ones((n, 1))
  
def A4_calc(m, n, k, B):
    I_n = np.ones(n, dtype=int)
    I_k = np.ones(k, dtype=int)
    I_k = I_k.reshape((1, k))
    I_n = I_n.reshape((1, n))

    E_n = np.eye(n)
    E_m = np.eye(m)
    
    B_ = np.diag(B)
    
    return np.kron(np.kron(I_n, B_), I_k), B.reshape((m, 1))

def A5_calc(m, n, k):
    I_n = np.ones(n, dtype=int)
    I_m = np.ones(m, dtype=int)
    I_m = I_m.reshape((1, m))
    I_n = I_n.reshape((1, n))

    E_n = np.eye(n)
    E_k = np.eye(k)
    
    return np.kron(np.kron(E_n, I_m), E_k), np.ones((k * n, 1))

def A6_calc(m, n, k):
    I_n = np.ones(n, dtype=int)
    I_m = np.ones(m, dtype=int)
    I_m = I_m.reshape((1, m))
    I_n = I_n.reshape((1, n))

    E_n = np.eye(n)
    E_m = np.eye(m)
    
    matrix = np.zeros((1, k))
    matrix[0,0] = 1
    matrix[0, n-1] = 1
    M_k = matrix
    
    matrix = np.zeros((1, m))
    matrix[0,0] = 1
    M_m = matrix
    
    return np.kron(np.kron(E_n, M_m), M_k), np.ones((n, 1))

def unbinarize_vector(x: np.array, decimals: int) -> np.array:
    binary_vec = np.array([1 / (2 ** (i + 1)) for i in range(decimals)])

    new_x = []
    for i in range(x.shape[0] // decimals):
        new_x.append( binary_vec.T @ x[i * decimals : (i + 1) * decimals] )
    return np.array(new_x)

def linear_to_QUBO(A: np.ndarray, b: np.ndarray, decimals: int) -> np.ndarray:
    binarian_vector = np.array([1 / (2 ** (i + 1)) for i in range(decimals)])
    binarian_matrix = binarian_vector[..., np.newaxis] @ binarian_vector[np.newaxis, ...]

    new_b = np.kron(b, binarian_vector)
    new_A = np.kron(A, binarian_matrix)

    return new_A + np.diag(new_b)

def remove_full_groups(B, distances):
    indices_of_full_groups = np.where(B == 9.)[0]
    indices_of_other_groups = np.where(B != 9.)[0]
    B_new = B[indices_of_other_groups]
    distances_new = np.delete(distances, indices_of_full_groups, axis = 0)
    distances_new = np.delete(distances_new, indices_of_full_groups, axis = 1)
    return B_new, distances_new, indices_of_full_groups.shape[0]

class LinearProblem():
    def __init__(self, A, b):
        self.A = A
        self.b = b
        print(b.shape)

    def add_equation_constraint(self, Q, v, weight):
        self.A += weight * Q.T @ Q
        self.b += -2 * weight * (Q.T @ v).T

    def add_inequality_to_one_constraint(self, Q, v_size: int, weight: float):
        L = np.ones((v_size, v_size))
        v = np.ones((v_size))

        print(self.A.shape, Q.shape, L.shape)
        self.A = np.block([[self.A + weight * (Q.T @ Q), weight * (Q.T @ L)], [weight * (L.T @ Q), weight * (L.T @ L)]])
        self.b = np.block([self.b - weight * (Q.T @ v), -weight * (L.T @ v)])

    def to_QUBO(self):
        return self.A + np.diag(self.b)

    
if __name__ == "__main__":
        
    file_path = 'task-2-nodes.csv'
    
    data = pd.read_csv(file_path, names=['name', 'count'])
    B = data['count'].to_numpy()

    file_path = 'task-2-adjacency_matrix.csv'

    df = pd.read_csv(file_path)

    distances = df.iloc[:, 1:].to_numpy()

    n = 15
    k = 15
    m = distances.shape[0]

    distances = np.where(distances == '-', 10000, distances)

    distances = distances.astype(float)

    B, distances, delta = remove_full_groups(B, distances)
    n -= delta
    m -= delta
    
    A1 = A1_calc(m, n, k, distances)    
    A2, B2 = A2_calc(m, n, k)
    A3, B3 = A3_calc(m, n, k, B)
    A4, B4 = A4_calc(m, n, k, B)
    A5, B5 = A5_calc(m, n, k)
    A6, B6 = A6_calc(m, n, k)

    H = LinearProblem(A1, np.zeros((1,A1.shape[1])))
    
    H.add_equation_constraint(A4, B4, 1)
    H.add_equation_constraint(A5, B5, 1)
    H.add_equation_constraint(A6, B6, 1)
    
    H.add_inequality_to_one_constraint(A2, B2.shape[0], 1)
    
    pad_width = H.b.shape[1] - A3.shape[1]
    pad_height = 0

    A3_new = np.pad(A3, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=1)
    
    H.add_inequality_to_one_constraint(A3_new, B3.shape[0], 1)
    
    H = H.to_QUBO()
    
    result = pq.solve(H)

    result_vector = np.array(result.vector)
    result_vector = result_vector[:A1.shape[0]]
    A1[A1 > 1000] = 0
    price = result_vector.T @ A1 @ result_vector
    print("price: ", price)
