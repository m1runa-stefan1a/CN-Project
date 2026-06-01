import numpy as np
import time
import matplotlib.pyplot as plt
# Importăm funcțiile din codul vostru original, fără să îl modificăm
from svd import svd_hand, svd_numpy_truncated

def test_ortogonalitate():
    print("=== TEST 4.3.1: Ortogonalitatea matricei U ===")
    m = 100
    k = 15
    # Generăm o matrice aleatoare care simulează matricea de contur
    A_test = np.random.rand(m, m)
    
    # Rulăm algoritmul vostru
    U, Sigma, Vt = svd_hand(A_test, k=k)
    
    # Calculăm U^T * U
    Ut_U = U.T @ U
    
    # Generăm matricea identitate I_k
    I_k = np.eye(k)
    
    # Calculăm eroarea (norma diferenței)
    eroare = np.linalg.norm(Ut_U - I_k)
    
    print(f"Dimensiune U: {U.shape}")
    print(f"Eroarea de ortogonalitate || U^T * U - I_k || este: {eroare:.15e}")
    if eroare < 1e-10:
        print("-> TEST TRECUT: Vectorii sunt perfect ortogonali (diferența e doar eroare de virgulă mobilă).")
    print("\n")

def test_timp_executie():
    print("=== TEST 4.3.2: Generare date pentru Graficul 1 ===")
    # Dimensiunile matricelor pe care le vom testa
    dimensiuni = [50, 100, 200, 300, 400]
    k = 15
    
    timpi_hand = []
    timpi_numpy = []
    
    for m in dimensiuni:
        A_test = np.random.rand(m, m)
        
        # Măsurăm timpul pentru SVD-ul vostru
        start = time.time()
        svd_hand(A_test, k=k)
        durata_hand = time.time() - start
        timpi_hand.append(durata_hand)
        
        # Măsurăm timpul pentru NumPy
        start = time.time()
        svd_numpy_truncated(A_test, k=k)
        durata_numpy = time.time() - start
        timpi_numpy.append(durata_numpy)
        
        print(f"Dimensiune {m}x{m} -> svd_hand: {durata_hand:.4f}s | numpy: {durata_numpy:.4f}s")
        
    # Desenăm graficul
    plt.figure(figsize=(8, 5))
    plt.plot(dimensiuni, timpi_hand, marker='o', label='svd_hand (Iterativ, k=15)', color='blue')
    plt.plot(dimensiuni, timpi_numpy, marker='s', label='NumPy (Direct QR)', color='red')
    
    plt.title('Timpul de execuție: SVD Propriu vs NumPy')
    plt.xlabel('Dimensiunea matricei (m x m)')
    plt.ylabel('Timp de execuție (secunde)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Salvăm graficul ca imagine, ca să îl puteți pune în documentație
    plt.savefig('grafic_timp_executie.png')
    print("\nGraficul a fost salvat ca 'grafic_timp_executie.png' în folderul curent.")
    plt.show()

if __name__ == "__main__":
    test_ortogonalitate()
    test_timp_executie()