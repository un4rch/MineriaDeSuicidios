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
###############################################################################################################################################
#                                                                                                                                             #
#                                                       Datos de entrada (preproceso)                                                         #
#                                                                                                                                             #
###############################################################################################################################################
soloPreproceso = False # True (solo preprocesar datos) | False (ademas de preprocesar, entrenar modelo o hacer predicciones)
#---- preprocessed_file != None ----------------------------------------------------------------------------------------------------------------
preprocessed_file = "50000instancias_prep.csv" # str (fichero datos preprocesados) | None (usar variable "unpreprocessed_file" para preprocesar)
#---- preprocessed_file == None ----------------------------------------------------------------------------------------------------------------
unpreprocessed_file = "50000instancias.csv" # Nombre del fichero con los datos SIN preprocesar
guardarPreproceso = "50000instancias_prep.csv" # str (fichero donde se guardaran los datos preprocesados) | None (no guardar preproceso)
doc2vec_model = "50000instancias_doc2vec.model" # str (usar un modelo doc2vec entrenado) | None (entrenar un modelo usando "doc2vec_vectors_size")
doc2vec_vectors_size = 1500 # Tamaño del vector doc2vec
pca_model = "50000instancias_pca.model" # str (usar un modelo pca entrenado) | None (entrenar un modelo usando "pca_dimensions")
pca_dimensions = 200 # Reduccion de dimensiones de los atributos (elegir los "n" mas representativos)
#----------------------------------------------------------------------------------------------------------------------------------------------

###############################################################################################################################################
#                                                                                                                                             #
#                      Elegir entre: entrenar un modelo de KMeans o hacer predicciones usando un modelo ya entrenado                          #
#                                                                                                                                             #
###############################################################################################################################################
train = True # True (entrenar un modelo de KMeans) | False (Realizar predicciones con los datos)
#----------------------------------------------------------------------------------------------------------------------------------------------

###############################################################################################################################################
#                                                                                                                                             #
#                            Entrenamiento (train == True): Generar un modelo KMeans configurando hiperparametros                             #
#                                                                                                                                             #
###############################################################################################################################################
test_size = 0.2 # Porcentaje cuantos datos se van a usar para train y para test (0.2 --> 20% test, 80% train)
n_clusters = 7 # Numero de clusters
maxIter = None # int (numero maximo de iteraciones como criterio de convergencia) | None (usar variable "tolerance" como criterio de convergencia)
tolerance = 1e-4 # float (usar un umbral como criterio de convergencia, asumiendo un error poco relevante)
centorids_init = "random_init" # str (modo de inicializacion de los centroides: random_init, space_division_init, separated_init)
p_minkowski = 7.5 # int/float (seleccionar que distancia se quiere usar: 1 (manhattan), 2 (euclidean), 7.5 (minkowski), "n" (cualquiera))
saveModeloKmeans = "50000instancias_kmeans_model.pkl" # str (nombre del fichero donde se va a guardar el modelo kmeans) | None (no guardar modelo)
saveMappingKmeans = "50000instancias_kmeans.map" # str (guardar el mapeo class-to-cluster) | None (no guardar mapeo)
#----------------------------------------------------------------------------------------------------------------------------------------------

###############################################################################################################################################
#                                                                                                                                             #
#                                                             Metricas y Testing                                                              #
#                                                                                                                                             #
###############################################################################################################################################
imprimirMetricas = True # True (Realizar pruebas con metricas)
#----------------------------------------------------------------------------------------------------------------------------------------------
saveMetricas = "50000instancias_metricas.txt" # str (fichero donde se van a guardar las metricas del algoritmo KMeans) | None (no guardar metricas)
numIteracionesCodos = None # 1 (Realizar el metodo de los codos y generar 1 grafica de inercias) |
                           # > 1 (Generar grafica donde se muestra con cuanta frecuencia los clusters han sido optimos) |
                           # None (no hacer metodo de los codos)
n_codos = 10 # int (Seleccionar rango de codos a utilizar: [1,n_codos])

###############################################################################################################################################
#                                                                                                                                             #
#          Barrido de hiperparametros: guardar las metricas de cada modelo para realizar comparativas y seleccionar el mejor modelo           #
#                                                                                                                                             #
###############################################################################################################################################
doTesting = False # True (realizar barrido de hiperparametros) | False (no realizar barrido de hiperparametros)
testing_dir = "pruebas" # str (directorio donde se van a guardar todas las metricas y modelos kmeans correspondientes) si doTesting == True
p_n_clusters = [2,7,8] # Lista de numero de clusters a probar
p_dists = [1,2,7.5] # Lista de tipos de distancias a probar
p_inits = ["random_init", "space_division_init", "separated_init"] # Lista de tipos de inicializaciones a probar
p_iters = [1e-4, 100] # Lista de criterios de convergencia a probar

###############################################################################################################################################
#                                                                                                                                             #
#                            Predicciones (If train == False): Usar un modelo KMeans para realizar predicciones                               #
#                                                                                                                                             #
###############################################################################################################################################
useModeloKmeans = "50000instancias_kmeans_model.pkl" # str (modelo de KMeans que se va a utilizar para hacer predicciones)
useMappingKmeans = "50000instancias_kmeans.map" # str (mapeo class-to-cluster para convertir "n" clusters al numero de clusters original) | 
                                                # None (no usar mapeo, y predecir con el numero de custers con el que ha sido entrenado el modelo KMeans)
output_prediction_file = "predicted.csv" # str (nombre fichero donde se van a guardar las predicciones) | None (no guardar predicciones e imprimirlas)
```
