import pandas as pd
import numpy as np
import pyqiopt as pq

file_path = 'task-1-stocks.csv'
def read_data(file_path):
    data = pd.read_csv(file_path)
    returns = pd.DataFrame(columns=data.columns)

    for col in data.columns:
        returns[col] = data[col].pct_change()  
    average_returns = returns.mean(axis=0)
    
    cov_matrix = data.cov()
    cov_matrix_np = np.array(cov_matrix)
    return cov_matrix_np, average_returns.to_numpy()

def unbinarize_vector(x: np.array, decimals: int) -> np.array:
    binary_vec = np.array([1 / (2 ** (i + 1)) for i in range(decimals)])

    new_x = []
    for i in range(x.shape[0] // decimals):
        new_x.append( binary_vec.T @ x[i * decimals : (i + 1) * decimals] )
    return np.array(new_x)

def add_equality_constraint(a: np.ndarray, b: float):
    quadratic_constraint = a[..., np.newaxis] @ a[np.newaxis, ...]
    linear_constraint = (-2 * b) * a
    return quadratic_constraint, linear_constraint

def linear_to_QUBO(A: np.ndarray, b: np.ndarray, decimals: int) -> np.ndarray:
    binarian_vector = np.array([1 / (2 ** (i + 1)) for i in range(decimals)])
    binarian_matrix = binarian_vector[..., np.newaxis] @ binarian_vector[np.newaxis, ...]

    new_b = np.kron(b, binarian_vector)
    new_A = np.kron(A, binarian_matrix)

    return new_A + np.diag(new_b)

def price(data, portfel):
    t = data.iloc[:,1].shape[0]
    p = [0] * t
    for i in range(0, t):
        p[i] = data.iloc[i,:].to_numpy() @ portfel
    return p

def risk(data, portfel):
    p = pd.DataFrame(price(data, portfel))
    r = p.pct_change()
    r_mean = r.mean(axis = 0)
    n = r.size
    return np.sqrt((((r - r_mean) ** 2 / (n - 1)).sum() * n).to_numpy())

if __name__ == "__main__":
    Q, r = read_data(file_path)
    A_sum, b_sum = add_equality_constraint(np.ones(r.shape), 1.)
    b = np.array([-r, b_sum, np.zeros(b_sum.shape)])
    A = np.array([np.zeros(A_sum.shape), A_sum, Q])
    weight_risk = 0.015 
    weight_sum = 0.1
    weight_return = 1

    weight = np.array([weight_return, weight_sum, weight_risk])
    H = linear_to_QUBO((A.T @ weight).T, b.T @ weight, 10)
    result = pq.solve(H)
    result_vec = unbinarize_vector(np.array(result.vector), 10)

    capital = 1e6
    portfolio = (capital * result_vec).astype(int)

    risk_res = risk(pd.read_csv(file_path), portfel)
    print("risk: ", risk_res)
    print("returns: ", result_vec.T @ r)
    print("sum: ", result_vec.sum())
    print("portfolio: ", ' '.join([str(e) for e in portfolio]), '\n')
    