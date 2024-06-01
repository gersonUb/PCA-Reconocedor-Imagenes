import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy.linalg as npAlg

"""
LAS FUNCIONES KNN ECHAS
"""

def KNNv1(Entrenamiento_datos : np.ndarray[np.float64], Entrenamiento_tipos : np.ndarray[np.float64], Test_datos : np.ndarray[np.float64], k : int):
    n = np.size(Test_datos, axis=0)
    m = np.size(Entrenamiento_tipos)
    Resultados = np.zeros(n)

    for i in range(n): #para cada imagen de test
        distancias = np.zeros(m)
        
        for j in range(m): #para cada imagen de entrenamiento
            distancias[j] = 1 - np.dot(Test_datos[i], Entrenamiento_datos[j])/(npAlg.norm(Test_datos[i])*npAlg.norm(Entrenamiento_datos[j])) #tomo la distancia entre la imagen de test y entrenamiento
        
        indices = np.argsort(distancias) #ordeno las distancias de menor a mayor. Indices guarda el orden de los indices ordenados (ej: los indices ordenados de [5, 2, 10] son [1, 0, 2])
        tipos = np.zeros(10)
        
        for j in range(k):
            tipos[Entrenamiento_tipos[indices[j]]] += 1 #me fijo los k mas cercanos y cuento que tipo es cada uno
        
        Resultados[i] = np.argmax(tipos) #guardo el tipo que tuvo mas cantidad en los k mas cercanos
    return Resultados

def KNNv2(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        # Calcular todas las similitudes coseno entre el punto de prueba y todos los puntos de entrenamiento
        similarities = 1 - np.dot(X_train, test_point) / (np.linalg.norm(X_train, axis=1) * np.linalg.norm(test_point))
        
        # Obtener los índices de los k vecinos más cercanos
        nearest_indices = np.argsort(similarities)[:k]
        
        # Obtener las etiquetas de los k vecinos más cercanos
        nearest_labels = y_train[nearest_indices]
        
        # Contar las ocurrencias de cada etiqueta
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        
        # Predecir la etiqueta más frecuente
        predicted_label = unique_labels[np.argmax(counts)]
        
        # Agregar la predicción a la lista de predicciones
        predictions.append(predicted_label)
    
    return np.array(predictions)

class_names = ["T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

X_train = np.loadtxt("data/X_train.csv", delimiter=",")
y_train = np.loadtxt("data/y_train.csv", delimiter=",").astype(int)
X_test = np.loadtxt("data/X_test.csv", delimiter=",")
y_test = np.loadtxt("data/y_test.csv", delimiter=",").astype(int)

#X_train y test contienen imagenes, y_train y test dice qué tipo de ropa es

print("1", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print("2", np.bincount(y_train), np.bincount(y_test))

X_newtrain, X_dev, y_newtrain, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print("3", np.bincount(y_newtrain), np.bincount(y_dev))
print("4", X_newtrain.shape, y_newtrain.shape, X_dev.shape, y_dev.shape)

resultados1 = KNNv1(X_newtrain, y_newtrain, X_dev, 5)

print("El algoritmo ha acertado para knn.v1", np.sum(resultados1 == y_dev),"de", y_dev.size, "veces")

resultados2 = KNNv2(X_newtrain, y_newtrain, X_dev, 5)

print("El algoritmo ha acertado para knn.v2", np.sum(resultados1 == y_dev),"de", y_dev.size, "veces")
