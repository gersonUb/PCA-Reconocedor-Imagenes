import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""
LA IDEA ES VER ESTE EXPERIMENTO CON LAS FUNCIONES
KNN Y PCA dadas por las librerias 
igualmente el experimento tarda un poquito pero es mucho mas rapido que el propuesto en el tp

"""

# Cargar los datos
X_train = np.loadtxt("data/X_train.csv", delimiter=",")
y_train = np.loadtxt("data/y_train.csv", delimiter=",").astype(int)
X_test = np.loadtxt("data/X_test.csv", delimiter=",")
y_test = np.loadtxt("data/y_test.csv", delimiter=",").astype(int)

# Parámetros a explorar
k_valores = [1, 3, 5, 7, 9]
p_valores = [10, 20, 30, 40, 50]  # Número de componentes principales

# Preparar la validación cruzada
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

mejor_exactitud = 0
mejor_k = None
mejor_p = None

# Exploración conjunta de hiperparámetros
for k in k_valores:
    for p in p_valores:
        accuracies = []
        
        for train_index, val_index in kf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
            # Aplicar PCA a los datos de entrenamiento del fold
            pca = PCA(n_components=p)
            X_train_fold_pca = pca.fit_transform(X_train_fold)
            X_val_fold_pca = pca.transform(X_val_fold)
            
            # Usar KNeighborsClassifier para predecir en el conjunto de validación
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_fold_pca, y_train_fold)
            y_val_pred = knn.predict(X_val_fold_pca)
            
            accuracy = accuracy_score(y_val_fold, y_val_pred)
            accuracies.append(accuracy)
        
        mean_accuracy = np.mean(accuracies)
        print(f"Promedio de exactitud para: k={k}, p={p}: {mean_accuracy:.4f}")
        
        if mean_accuracy > mejor_exactitud:
            mejor_exactitud = mean_accuracy
            mejor_k = k
            mejor_p = p

print(f"Mejor k: {mejor_k}, mejor p: {mejor_p}, mejor exactitud: {mejor_exactitud:.4f}")

# Entrenar con todos los datos de entrenamiento usando los mejores hiperparámetros
pca = PCA(n_components=mejor_p)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=mejor_k)
knn.fit(X_train_pca, y_train)
y_test_pred = knn.predict(X_test_pca)

test_exactitud = accuracy_score(y_test, y_test_pred)
print(f"Exactitud en el conjunto de prueba con el mejor k y p: {test_exactitud:.4f}")


import matplotlib.pyplot as plt

# Función para mostrar imágenes con sus etiquetas predichas
def mostrar_imagenes_con_prediccion(imagenes, etiquetas_reales, etiquetas_predichas, clases, n_imagenes=5):
    fig, axs = plt.subplots(2, n_imagenes, figsize=(15, 6))
    for i in range(n_imagenes):
        # Mostrar la imagen
        axs[0, i].imshow(imagenes[i], cmap='gray')
        axs[0, i].set_title(f'Real: {clases[etiquetas_reales[i]]}')
        axs[0, i].axis('off')
        # Mostrar la etiqueta predicha
        axs[1, i].text(0.5, 0.5, f'Predicción: {clases[etiquetas_predichas[i]]}', ha='center', va='center', fontsize=12)
        axs[1, i].axis('off')
    plt.show()

# Mostrar una muestra de imágenes con sus etiquetas predichas
mostrar_imagenes_con_prediccion(X_test.reshape(-1, 28, 28), y_test, y_test_pred, clases={0: 'Camiseta', 1: 'Pantalón', 2: 'Suéter', 3: 'Vestido', 4: 'Abrigo', 5: 'Sandalia', 6: 'Camisa', 7: 'Zapatilla', 8: 'Bolso', 9: 'Bota'})
