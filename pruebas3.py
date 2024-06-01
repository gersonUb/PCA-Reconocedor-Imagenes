import numpy as np
import matplotlib.pyplot as plt

def calcular_autovalores_autovectores(CA, num_iter=20000, eps=1e-24):
    n = CA.shape[0]
    autovalores = []
    autovectores = []

    for i in range(n):
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)
        l = 0
        for j in range(num_iter):
            v_new = CA @ v
            v_new = v_new / np.linalg.norm(v_new)
            l_new = v_new @ CA @ v_new
            if np.abs(l_new - l) < eps:
                break
            v = v_new
            l = l_new
        autovalores.append(l)
        autovectores.append(v)

    return np.array(autovalores), np.array(autovectores).T

def PCA(A):
    A_mean = np.mean(A, axis=0)
    A_centered = A - A_mean
    
    n = A_centered.shape[1]
    CA = (A_centered.T @ A_centered) / (n - 1)
    
    autovalores, autovectores = calcular_autovalores_autovectores(CA)
    
    indices_orden = np.argsort(autovalores)[::-1]
    autovalores = autovalores[indices_orden]
    autovectores = autovectores[:, indices_orden]
    
    return autovalores, autovectores, A_mean

A = np.random.rand(50, 25)

autovalores, autovectores, A_mean = PCA(A)

A_transformado = (A - A_mean) @ autovectores

nombres_caracteristicas = [f'Característica {i+1}' for i in range(A.shape[1])]

coeficientes_componentes = autovectores.T

k = 5

for i in range(k):
    plt.figure(figsize=(10, 6))
    plt.bar(nombres_caracteristicas, coeficientes_componentes[i, :])
    plt.xlabel('Características Originales')
    plt.ylabel(f'Coeficientes del Componente Principal {i+1}')
    plt.title(f'Coeficientes del Componente Principal {i+1} vs. Características Originales')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()



