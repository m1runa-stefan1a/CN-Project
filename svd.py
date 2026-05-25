import numpy as np

def svd_1d(A, epsilon=1e-10, max_iter=100):
    """ One-dimensional SVD using Power Iteration. """
    n, m = A.shape
    # Handle zero matrix gracefully to avoid division by zero / NaN
    if np.linalg.norm(A) < epsilon:
        return np.zeros(n), 0.0, np.zeros(m)
    
    x = np.random.randn(m)
    x = x / np.linalg.norm(x)
    
    for _ in range(max_iter):
        last_x = x
        # Power iteration step to find the dominant eigenvector of A^T * A
        # Suppress warnings that can occur with certain BLAS backends
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            x = A.T @ (A @ x)
        # Check for numerical blow-up
        if not np.all(np.isfinite(x)):
            return np.zeros(n), 0.0, np.zeros(m)
        norm_x = np.linalg.norm(x)
        if norm_x < epsilon:
            return np.zeros(n), 0.0, np.zeros(m)
        x = x / norm_x
        # Check for convergence
        if np.abs(np.dot(x, last_x)) > 1 - epsilon:
            break
            
    v = x
    u = A @ v
    sigma = np.linalg.norm(u)
    if sigma > 0:
        u = u / sigma
    return u, sigma, v

def svd_hand(A, k=10):
    """ SVD by hand for top k singular values using matrix deflation. """
    A_copy = np.copy(A)
    n, m = A.shape
    U = np.zeros((n, k))
    Sigma = np.zeros(k)
    Vt = np.zeros((k, m))
    
    for i in range(k):
        u, sigma, v = svd_1d(A_copy)
        U[:, i] = u
        Sigma[i] = sigma
        Vt[i, :] = v
        # Deflate the matrix to find the next largest singular value
        A_copy = A_copy - sigma * np.outer(u, v)
        
    return U, Sigma, Vt