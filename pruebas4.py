import numpy as np
import matplotlib.pyplot as plt
"""
PRUEBO CON UN METODO DE POTENCIA EN PYTHON
Y CON ENTRADA DE INFORMACION (MATRIZ) DE 50X25 
Y ESA CANTIDAD DE DATOS MI EXPERIMENTO ANDA BIEN
"""
def calcular_autovalores_autovectores(CA, num_iter=20000, eps=1e-24):
    # Función para encontrar los autovalores y autovectores de CA utilizando el método de la potencia
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
    # Paso 1: Restar el promedio de cada columna de A
    A_mean = np.mean(A, axis=0)
    A_centered = A - A_mean
    
    # Paso 2: Construir la matriz de covarianza CA
    n = A_centered.shape[1]
    CA = (A_centered.T @ A_centered) / (n - 1)
    
    # Paso 3: Encontrar los autovalores y autovectores de CA
    autovalores, autovectores = calcular_autovalores_autovectores(CA)
    
    # Paso 4: Construir la matriz de cambio de base V con los autovectores
    V = autovectores

    return V, A_mean

# Ejemplo de uso:

# Suponiendo que tienes una matriz de datos A
A = np.random.rand(50, 25)  # Ejemplo de matriz de datos de tamaño 50x25

# Aplicar PCA a la matriz de datos A
V, A_mean = PCA(A)

# V es la matriz de cambio de base que puedes usar para transformar tus datos

# V es la matriz de cambio de base que puedes usar para transformar tus datos
print("Matriz de cambio de base V:")
print(V)

# Para mostrar los resultados de la transformación, puedes usar la siguiente línea:
A_transformado = (A - A_mean) @ V
print("Datos transformados:")
print(A_transformado)


# Calcular la varianza explicada
explained_variance = np.var(A_transformado, axis=0) / np.sum(np.var(A_transformado, axis=0))

# Calcular la varianza explicada acumulativa
varianza_explicada_acumulativa = np.cumsum(explained_variance)

# Graficar la varianza explicada acumulativa
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(varianza_explicada_acumulativa) + 1), varianza_explicada_acumulativa, marker='o', linestyle='-')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada Acumulativa')
plt.title('Varianza Explicada Acumulativa por Componentes Principales')
plt.ylim(0, 1)  # Corregir el rango del eje y para que vaya de 0 a 1
plt.grid(True)
plt.show()

