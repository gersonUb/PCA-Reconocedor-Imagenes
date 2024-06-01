import sys
from sklearn.decomposition import PCA
import numpy as np
import time 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy.linalg as npAlg

"""
LA IDEA ES COMO ES MI FUNCION KNN RESPECTO AL DE LA LIBRERIA

k=1, Exactitud promedio=0.8010
k=3, Exactitud promedio=0.8080
k=5, Exactitud promedio=0.8148
k=7, Exactitud promedio=0.8056
k=9, Exactitud promedio=0.8058
Mejor valor de k: 5
Precisión en el conjunto de prueba con k=5: 0.8140

EL SEGUNDO MODELO CON MI FUNCION KNN TARDA DEMASIADO EN COMPILAR PERO EN FIN FUNCIONA Y ME DA COMO RESULTADO:
con mi funcion KNN
k=1, Exactitud promedio=0.8036
k=3, Exactitud promedio=0.8054
k=5, Exactitud promedio=0.7950
k=7, Exactitud promedio=0.7940
k=9, Exactitud promedio=0.7894
Mejor valor de k: 3

"""

# Rutas de los archivos CSV
X_train = np.loadtxt("data/X_train.csv", delimiter=",")
y_train = np.loadtxt("data/y_train.csv", delimiter=",").astype(int)
X_test = np.loadtxt("data/X_test.csv", delimiter=",")
y_test = np.loadtxt("data/y_test.csv", delimiter=",").astype(int)

print("ejercicio 3)b)")

# Definir valores de k a probar
k_values = [1, 3, 5, 7, 9]

# Inicializar lista para almacenar las medidas de exactitud promedio
exactitud_promedio = []

# Definir el número de folds para la validación cruzada
num_folds = 5

# Crear objeto StratifiedKFold
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Iterar sobre los valores de k
for k in k_values:
    fold_accuracies = []
    
    # Validación cruzada
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        # Crear y entrenar el modelo k-NN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_fold, y_train_fold)
        
        # Predecir sobre el conjunto de validación
        y_val_pred = knn.predict(X_val_fold)
        
        # Calcular la precisión y agregarla a la lista de precisiones
        accuracy = accuracy_score(y_val_fold, y_val_pred)
        fold_accuracies.append(accuracy)
    
    # Calcular la precisión promedio para el valor actual de k
    avg_accuracy = np.mean(fold_accuracies)
    exactitud_promedio.append(avg_accuracy)
    print(f'k={k}, Exactitud promedio={avg_accuracy:.4f}')

# Encontrar el valor de k con la mejor precisión promedio
best_k = k_values[np.argmax(exactitud_promedio)]
print(f'Mejor valor de k: {best_k}')

# Graficar las precisiones promedio para cada valor de k
plt.plot(k_values, exactitud_promedio, marker='o')
plt.xlabel('Valor de k')
plt.ylabel('Precisión Promedio')
plt.title('Precisión Promedio vs Valor de k')
plt.show()

# Entrenar el modelo final con el mejor valor de k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_test_pred = knn_best.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Precisión en el conjunto de prueba con k={best_k}: {test_accuracy:.4f}')

#########################################################################################################################################
#########################################################################################################################################

print("con mi funcion KNN")

def KNN(X_train, y_train, X_test, k):
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

# Definir valores de k a probar
k_values = [1, 3, 5, 7, 9]

# Inicializar lista para almacenar las medidas de exactitud promedio
exactitud_promedio = []

# Definir el número de folds para la validación cruzada
num_folds = 5

# Crear objeto StratifiedKFold
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Iterar sobre los valores de k
for k in k_values:
    fold_accuracies = []
    
    # Validación cruzada
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        # Predecir sobre el conjunto de validación usando la función KNN
        y_val_pred = KNN(X_train_fold, y_train_fold, X_val_fold, k)
        
        # Calcular la precisión y agregarla a la lista de precisiones
        accuracy = accuracy_score(y_val_fold, y_val_pred)
        fold_accuracies.append(accuracy)
    
    # Calcular la precisión promedio para el valor actual de k
    avg_accuracy = np.mean(fold_accuracies)
    exactitud_promedio.append(avg_accuracy)
    print(f'k={k}, Exactitud promedio={avg_accuracy:.4f}')

# Encontrar el valor de k con la mejor precisión promedio
best_k = k_values[np.argmax(exactitud_promedio)]
print(f'Mejor valor de k: {best_k}')

