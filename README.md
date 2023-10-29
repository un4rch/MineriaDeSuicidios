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
## Si queremos solamente preprocesar:
```
soloPreproceso: True
preprocessed_file: None
unpreprocessed_file: str(”<nombre-fichero>”)
pca_dimensions: int(<num-dimensiones>)
guardarPreproceso:
∗ ”<nombre-fichero>”: para guardar el preproceso
∗ None: para no guardar el preproceso
```
## Si queremos entrenar un modelo:
```
soloPreproceso: False
train: True
```
- Si los datos no están preprocesados:
```
preprocessed_file: None
unpreprocessed_file: str(”<nombre-fichero>”)
pca_dimensions: int(<num-dimensiones>)
guardarPreproceso:
   - str(”<nombre-fichero>”): para guardar el preproceso
   - None: para no guardar el preproceso
```
- Si los datos si están preprocesados:
```
preprocessed_file: str(”<nombre-fichero>”)
unpreprocessed_file: None
pca_dimensions: None
guardarPreproceso:
   - str(”<nombre-fichero>”): para guardar el preproceso
   - None: para no guardar el preproceso
n_clusters = int(<num-clusters>)
maxIter:
   - int(<max-iter>): para definir un numero tope de iteraciones para recalcular centroides
   - None: para utilizar una tolerancia en vez de un numero de iteraciones fijo
tolerance:
   - float(<max-iter>)): tolerancia que indica cuando parar de recalcular clusters (por defecto: 1e-4)
   - None: para usar un numero de iteraciones en vez de la tolerancia
centorids_init = str(”<inicializacion>”) (valores permitidos: random init, space division init, separated init)
p_minkowski = float(<max-iter>)
   - 1: Manhattan
   - 2: Euclidean
   - 7.5: Minkowski
   - Se pueden probar otros valores
test_size = float(<test size>) (test size ∈ [0, 1], nomalmente: 0.2)
saveModeloKmeans:
   - str(”<nombre-fichero>”): para guardar el modelo de kmeans
   - None: para no guardar el preproceso
imprimirMetricas:
   - True: para imprimir metricas durante el entrenamiento
   - False: para no imprimir metricas durante el entrenamiento
n_codos = int(<n codos>) (para elegir el numero optimo de clusters con el metodo del codo)
```
## Si queremos predecir clusters de un conjunto de datos:
```
– soloPreproceso: False
– train: False
```
- Si los datos no están preprocesados:
```
preprocessed_file: None
unpreprocessed_file: str(”<nombre-fichero>”)
doc2vec_model = str(”<nombre-fichero>”)
pca_model = str(”<nombre-fichero>”)
pca_dimensions: None
guardarPreproceso:
   - str(”<nombre-fichero>”): para guardar el preproceso
   - None: para no guardar el preproceso
```
- Si los datos si están preprocesados:
```
preprocessed_file: str(”<nombre-fichero>”)
unpreprocessed_file: None
pca_dimensions: None
```
```
guardarPreproceso:
   - str(”<nombre-fichero>”): para guardar el preproceso
   - None: para no guardar el preproceso
useModeloKmeans: str(”<nombre-fichero>”) (el modelo de kmeans que hemos generado y guardado durante el entrenamiento)
output_prediction_file: str(”<nombre-fichero>”) (para guardar las predicciones)
```
