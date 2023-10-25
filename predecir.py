from kmeans import KMeans
import pickle
import pandas as pd
import numpy as np
import sys
from preprocessor import Preprocessor

with open("kmeans_model.pkl", "rb") as file:
    kmeans = pickle.load(file)

dtPinnata = pd.read_csv(sys.argv[1])
x,y = np.asarray(dtPinnata["text"]),np.asarray(dtPinnata["class"])

label_map = {cat:index for index,cat in enumerate(np.unique(y))}
y = np.asarray([label_map[l] for l in y])

preprocessor = Preprocessor()
x_prep,y_prep = preprocessor.doc2vec(x, y, pca_dimensions=200)


assigned_labels = kmeans.predict(x_prep)

print(assigned_labels)

if not pd.isnull(y).all():
    aciertos = 0
    errores = 0
    for idx,label in enumerate(y):
        if y_prep[idx] == label:
            aciertos += 1
        else:
            errores += 1
    print(f"Aciertos: {aciertos}")
    print(f"Errores: {errores}")
    print("De momento no es fiable ya que las etiquetas se inician aleatoriamente, habria que cambiar la etiqueta de uno a todos")