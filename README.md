# Clonar el repositorio:
```
git clone https://github.com/un4rch/MineriaDeSuicidios.git
```
# Ejecutar la aplicación:
```
python clustering.py
```
# Configurar el comportamiento del programa:
Para configurar el comportamiento del programa, hay definidas unas variables las cuales debemos modificar su valor:
```
#--------------------------------------------------------#
# Varibales para configurar el comportamiento del script #
#--------------------------------------------------------#
# Preproceso
# ----------
soloPreproceso = False
preprocessed_file = "50000instancias_prep.csv"
# If preprocessed_file == None
unpreprocessed_file = "50000instancias.csv"
guardarPreproceso = "50000instancias_prep.csv"
pca_dimensions = 200
# If preprocessed_file not None
# Se usa la variable preprocessed_file
# If soloPreproceso == True
# No se ejecuta nada mas de las variables de abajo

train = True # True: train, False: predict
# Entrenamiento (If train == True)
# --------------------------------
n_clusters = 8 # 3,7,8
maxIter = None #100
tolerance = 1e-4 # If maxIter == None, stop when has converged using this tolerance
centorids_init = "random_init" #random_init, space_division_init, separated_init
p_minkowski = 7.5
test_size = 0.2 #20%
saveModeloKmeans = "50000instancias_kmeans_model.pkl" #None if you do not want to save model to predict later
saveMappingKmeans = "50000instancias_kmeans.map"
imprimirMetricas = True
saveMetricas = "50000instancias_metricas.csv"
# If imprimirMetricas == True
n_codos = None # None if not want to make elbow method
numIteracionesCodos = None

# Predicciones (If train == False)
# --------------------------------
useModeloKmeans = "kmeans_model.pkl"
useMappingKmeans = "50000instancias_kmeans.map"
doc2vec_model = "50000instancias_doc2vec.model" # None to train, else use trained model to predict
pca_model = "50000instancias_pca.model"
output_prediction_file = "predicted.csv"

# Testing
#--------
doTesting = False
testing_dir = "pruebas"
p_n_clusters = [2,7,8]
p_dists = [1,2,7.5]
p_inits = ["random_init", "space_division_init", "separated_init"]
p_iters = [1e-4, 100]
```
