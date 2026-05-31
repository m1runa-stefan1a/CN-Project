import numpy as np
import time

def svd_1d(A, epsilon=1e-10, max_iter=100):
    n, m = A.shape
    if np.linalg.norm(A) < epsilon:
        return np.zeros(n), 0.0, np.zeros(m)
    
    x = np.random.randn(m)
    x = x / np.linalg.norm(x)
    
    for _ in range(max_iter):
        last_x = x
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            x = A.T @ (A @ x)
        if not np.all(np.isfinite(x)):
            return np.zeros(n), 0.0, np.zeros(m)
        norm_x = np.linalg.norm(x)
        if norm_x < epsilon:
            return np.zeros(n), 0.0, np.zeros(m)
        x = x / norm_x
        if np.abs(np.dot(x, last_x)) > 1 - epsilon:
            break
            
    v = x
    u = A @ v
    sigma = np.linalg.norm(u)
    if sigma > 0:
        u = u / sigma
    return u, sigma, v

def svd_hand(A, k=10):
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
        A_copy = A_copy - sigma * np.outer(u, v)
        
    return U, Sigma, Vt

def svd_numpy_truncated(A, k=10):
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], Sigma[:k], Vt[:k, :]

def compare_svd_efficiency(iterations=100, k=15):
    print(f"--- Rulare test de eficiență ({iterations} repetiții, k={k}) ---")
    
    # Generăm o matrice aleatoare de 100x100
    test_matrix = np.random.rand(100, 100)
    
    # 1. Test pentru SVD-ul manual
    start_hand = time.time()
    for _ in range(iterations):
        _, _, _ = svd_hand(test_matrix, k)
    end_hand = time.time()
    time_hand = end_hand - start_hand
    
    # 2. Test pentru SVD-ul din NumPy
    start_np = time.time()
    for _ in range(iterations):
        _, _, _ = svd_numpy_truncated(test_matrix, k)
    end_np = time.time()
    time_np = end_np - start_np
    
    print(f"Timp total SVD Manual (Power Iteration): {time_hand:.4f} secunde")
    print(f"Timp total SVD NumPy (LAPACK/C): {time_np:.4f} secunde")
    if time_np > 0:
        print(f"🚀 NumPy este de {time_hand / time_np:.1f} ori mai rapid!")

if __name__ == "__main__":
    compare_svd_efficiency()