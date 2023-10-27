from kmeans import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import sys
import numpy as np
from preprocessor import Preprocessor
import os
import pickle
import matplotlib.pyplot as plt

#Opciones:
#dar un diccionario de numeros y labels "oficial para cambiarlos"
#gudardar fichero preprocesado o usar uno existente

#--------------------------------------------------------#
# Varibales para configurar el comportamiento del script #
#--------------------------------------------------------#
# Preproceso
# ----------
soloPreproceso = False
preprocessed_file = None
# If preprocessed_file == None
unpreprocessed_file = "100lineas.csv"
guardarPreproceso = "x_prep.csv"
pca_dimensions = 200
# If preprocessed_file not None
# Se usa la variable preprocessed_file
# If soloPreproceso == True
# No se ejecuta nada mas de las variables de abajo

train = False # True: train, False: predict
# Entrenamiento (If train == True)
# --------------------------------
n_clusters = 3
maxIter = None
tolerance = 1e-4 # If maxIter == None, stop when has converged using this tolerance
centorids_init = "random_init" #random_init, space_division_init, separated_init
p_minkowski = 2
saveModeloKmeans = "kmeans_model.pkl" #None if you do not want to save model to predict later
imprimirMetricas = True
# If imprimirMetricas == True
n_codos = 10 # None if not want to make elbow method

# Predicciones (If train == False)
# --------------------------------
useModeloKmeans = "kmeans_model.pkl"
doc2vec_model = "reddit_suicide_depression1500.model" # None to train, else use trained model to predict
output_prediction_file = "predicted.csv"

"""
# Fichero que representa las asignaciones oficiales tras ver las asignaciones numericas
assignLabels = None # {0: "depresion", 1: "" 2: "", etc...}
#test_size = 0.2 #20%
"""

def saveAssignedPredictions(filename, assigned_labels):
    with open(filename, "w") as file:
        writter = csv.writer(file)
        for post,label in assigned_labels.items():
            if post in vectors_list:
                idx = vectors_list.index(post)
                """if assignLabels:
                    writter.writerow([x[idx], assignLabels[label]])
                else:
                    writter.writerow([x[idx], label])"""
                writter.writerow([x[idx], label])

def metodo_codo(dataset, num_codos):
    sum_of_squared_distances = []
    for k in range(1, num_codos):
        kmeans = KMeans(dataset, n_clusters, maxIter, centorids_init, p_minkowski, tolerance)
        kmeans.fit()
        sum_of_squared_distances.append(kmeans.inertia)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_codos), sum_of_squared_distances, marker='o', linestyle='-', color='b')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Suma de distancias al cuadrado')
    plt.title('Método del codo para seleccionar k')
    plt.grid(True)
    plt.savefig(f'metodo_{num_codos}_codos.png', format='png')
    plt.close()
    k = np.argmin(sum_of_squared_distances[1:]) + 1
    return k

def load_dataset(filename):
    dataset = pd.read_csv(filename)
    return np.asarray(dataset["text"]),np.asarray(dataset["class"])

if __name__ == "__main__":
    vectors_list = None
    if preprocessed_file == None:
        print("Preproceso")
        print("----------")
        # Cargar el fichero de datos
        x,y = load_dataset(unpreprocessed_file)
        preprocessor = Preprocessor()
        if doc2vec_model:
            x_prep,y_prep,doc2vec_model = preprocessor.doc2vec(x, y, pca_dimensions=pca_dimensions, doc2vec_model=doc2vec_model)
        else:
            x_prep,y_prep,doc2vec_model = preprocessor.doc2vec(x, y, pca_dimensions=pca_dimensions, doc2vec_model=None)
        doc2vec_model.save(unpreprocessed_file.split(".")[1]+"_doc2vec.model")
        vectors_list = x_prep.tolist()
        vectors_list = [tuple(point) for point in vectors_list]
        print(f"[*] Preproceso listo")
        if guardarPreproceso != None:
            with open(guardarPreproceso, "w") as file:
                writer = csv.writer(file)
                writer.writerow(["text","class"])
                for idx,point in enumerate(vectors_list):
                    writer.writerow([point,y[idx]])
            print(f"    Fichero guardado: {guardarPreproceso}")
            print(f"    Fichero guardado: {unpreprocessed_file.split('.')[1]+'_doc2vec.model'}")
            print()
        if soloPreproceso:
            sys.exit(0)
    else:
        if not os.path.exists(preprocessed_file):
            print(f"Error: {preprocessed_file} not found")
            sys.exit(1)
        x_prep,y_prep = load_dataset(preprocessed_file)
        vectors_list = [eval(point) for point in x_prep]

    if useModeloKmeans:
        with open(useModeloKmeans, "rb") as file:
            kmeans = pickle.load(file)
    else:
        kmeans = KMeans(vectors_list, n_clusters, maxIter, centorids_init, p_minkowski, tolerance)
    
    if train: # Do train
        # Separar los datos en 2 conjuntos, train y test
        #x_train,x_test = train_test_split(vectors_list, test_size=test_size)
        print("[*] Entrenando kmeans...")
        print()
        centroids, clusters = kmeans.fit()

        # Elegir el numero de clusters optimo con el metodo de los codos
        if imprimirMetricas:
            n_clusters_optimo = metodo_codo(vectors_list, n_codos)
            print("Metricas")
            print("--------")
            print(f"[*] Numero optimo de clusters (elbow method): {n_clusters_optimo}")
            print(f"    Imagen guardada: metodo_{n_codos}_codos.png")
            print(f"[*] SSE (Sum of Squared Errors): {kmeans.inertia}")
            # TODO poner mas metricas
            # TODO comparar con kmeans de sklearn
            # TODO grafica de PCA usando la variable "pca_dimensions" (copia-pega de la entrega 1)
            print()
    
        assigned_labels = kmeans.assign_numeric_labels(clusters)
        #saveAssignedPredictions("train_labels_assigned.csv", assigned_labels)
        if saveModeloKmeans:
            with open(saveModeloKmeans, "wb") as file:
                pickle.dump(kmeans, file)
    else: # Do predict
        print("Predicciones")
        print("------------")
        assigned_labels = kmeans.predict(vectors_list)
        print(assigned_labels)
        #saveAssignedPredictions("test_labels_assigned.csv", assigned_labels)
    
    if output_prediction_file:
        with open(output_prediction_file, "w") as file:
            writter = csv.writer(file)
            for post,label in assigned_labels.items():
                if post in vectors_list:
                    idx = vectors_list.index(post)
                    """if assignLabels:
                        writter.writerow([x[idx], assignLabels[label]])
                    else:
                        writter.writerow([x[idx], label])"""
                    writter.writerow([x[idx], label])



"""
# mapping
labels_frecuency = {label: [0,0,0] for label in set(y)}
for post,label in assigned_labels.items():
    if post in vectors_list:
        idx = vectors_list.index(post)
        original_label = y[idx]
        labels_frecuency[original_label][label] += 1
        #print(original_label)
        #print(label)
print(labels_frecuency)
new_assigned_labels = {}
for post,label in assigned_labels.items():
    for original_label in labels_frecuency:
        idx = labels_frecuency[original_label].index(max(labels_frecuency[original_label]))
        if label == idx:
            new_assigned_labels[label] = original_label
print(new_assigned_labels)
"""
