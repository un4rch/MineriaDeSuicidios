from kmeans import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import sys
import numpy as np
from preprocessor import Preprocessor
import os
import pickle

#Opciones:
#dar un diccionario de numeros y labels "oficial para cambiarlos"
#gudardar fichero preprocesado o usar uno existente

n_clusters = 3
#assignLabels = {0: "depresion", 1: "" 2: "", etc...}
assignLabels = {}
preprocessed_file = "x_prep.csv"

def saveAssignedPredictions(filename, assigned_labels):
    with open(filename, "w") as file:
        writter = csv.writer(file)
        for post,label in asigned_labels.items():
            if post in vectors_list:
                idx = vectors_list.index(post)
                if assignLabels:
                    writter.writerow([x[idx], assignLabels[label]])
                else:
                    writter.writerow([x[idx], label])

dtPinnata = pd.read_csv(sys.argv[1])
x,y = np.asarray(dtPinnata["text"]),np.asarray(dtPinnata["class"])

vectors_list = []
if not os.path.exists("x_prep.csv"):
    preprocessor = Preprocessor()
    x_prep,y_prep = preprocessor.word2vec(x, y, pca_dimensions=200)
    #vectors_list = tuple([float(point) for point in x_prep])
    #vectors_list = [[point] for point in x_prep]
    vectors_list = x_prep.tolist()
    vectors_list = [tuple(point) for point in vectors_list]
else:
    vectors_list = []
    with open("x_prep.csv", "r") as file:
        for line in file.read().splitlines():
            line = [float(point) for point in line.split(",")]
            vectors_list.append(tuple(line))

x_train,x_test = train_test_split(vectors_list, test_size=0.2)
kmeans = KMeans(x_train, n_clusters)
centroids, clusters = kmeans.fit()
asigned_labels = kmeans.assign_numeric_labels(clusters)

saveAssignedPredictions("train_labels_asigned.csv", asigned_labels)

asigned_labels = kmeans.predict(x_test)

saveAssignedPredictions("test_labels_asigned.csv", asigned_labels)

with open("kmeans_model.pkl", "wb") as file:
    pickle.dump(kmeans, file)

"""
# mapping
labels_frecuency = {label: [0,0,0] for label in set(y)}
for post,label in asigned_labels.items():
    if post in vectors_list:
        idx = vectors_list.index(post)
        original_label = y[idx]
        labels_frecuency[original_label][label] += 1
        #print(original_label)
        #print(label)
print(labels_frecuency)
new_asigned_labels = {}
for post,label in asigned_labels.items():
    for original_label in labels_frecuency:
        idx = labels_frecuency[original_label].index(max(labels_frecuency[original_label]))
        if label == idx:
            new_asigned_labels[label] = original_label
print(new_asigned_labels)
"""